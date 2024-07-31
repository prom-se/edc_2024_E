#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <algorithm>
#include <array>
#include "include/serial_driver.hpp"
#include "include/board.hpp"

namespace edc
{
    class Slover
    {
    public:
        Slover(std::string cam_name = "/dev/video2") : src_{cv::Mat(cv::Size(1280, 720), CV_8UC3)}
        {
            cam_name_ = cam_name;
            serial_ = std::make_shared<VisionSerial>("/dev/ttyACM0", 115200);
            cam_ = cv::VideoCapture(cam_name_, cv::CAP_V4L2);
            board_ = std::make_shared<edc::Board>();
            camera_stream_ = std::make_shared<std::thread>(&edc::Slover::get_img, this);
            serial_driver_ = std::make_shared<std::thread>(&edc::Slover::serial_driver, this);
            board_finder_ =
                std::make_shared<std::thread>([this]() -> void
                                              {
            while (true)
            {
                while (!cam_.isOpened())
                {
                    cam_.open(cam_name_, cv::CAP_V4L2);
                }
                find_board();
            } });
        };
        ~Slover()
        {
            if (camera_stream_->joinable())
            {
                camera_stream_->join();
            }
            if (serial_driver_->joinable())
            {
                serial_driver_->join();
            }
            if (board_finder_->joinable())
            {
                board_finder_->join();
            }
        };

    private:
        void get_img()
        {
            cam_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cam_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cam_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            cam_.set(cv::CAP_PROP_FPS, 60);
            cam_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
            cam_.set(cv::CAP_PROP_EXPOSURE, 200);
#ifdef DEBUG
            // cv::namedWindow("src");
            cv::namedWindow("debug");
            cv::namedWindow("show");
            cv::createTrackbar("h_low", "debug", low, 255);
            cv::createTrackbar("s_low", "debug", low + 1, 255);
            cv::createTrackbar("v_low", "debug", low + 2, 255);
            cv::createTrackbar("h_high", "debug", high, 255);
            cv::createTrackbar("s_high", "debug", high + 1, 255);
            cv::createTrackbar("v_high", "debug", high + 2, 255);
#endif
            while (true)
            {
                cam_.read(src_);
                if (!src_.empty())
                {
#ifdef DEBUG
                    // cv::imshow("src", src_);
                    cv::waitKey(1);
#endif
                }
            }
        };

        void serial_driver()
        {
            while (true)
            {
                try
                {
                    uint8_t chesses[9];
                    float angle;
                    VisionMsg msg;
                    msg.head = 0xA5;
                    serial_->vision_update(msg);
                }
                catch (std::exception &ex)
                {
                }
            }
        };

        void find_board()
        {
            std::thread chess_finder(std::bind(&edc::Board::detect_chess,board_),src_);
            // 预处理
            cv::Mat show = src_.clone();
            cv::Mat hsv(cv::Mat::zeros(720, 1280, CV_8UC1));
            cv::Mat bin(cv::Mat::zeros(720, 1280, CV_8UC1));
            cv::Mat draw(cv::Mat::zeros(720, 1280, CV_8UC1));
            cv::cvtColor(src_, hsv, cv::COLOR_BGR2HSV_FULL);
            cv::inRange(hsv, cv::Scalar(low[0], low[1], low[2]), cv::Scalar(high[0], high[1], high[2]), bin);
            cv::threshold(bin, bin, 0, 255, cv::THRESH_BINARY);
            cv::morphologyEx(bin.clone(), bin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::array<cv::Point2f, 4> pts;
            for (auto &contour : contours)
            {
                // 面积筛选
                auto rect = cv::minAreaRect(contour);
                if (rect.size.area() > 600 * 600)
                {
                    continue;
                }
                if (rect.size.area() < 300 * 300)
                {
                    continue;
                }

                // 棋盘寻找
                draw = cv::Mat::zeros(720, 1280, CV_8UC1);
                std::vector<cv::Point2f> pts;
                std::vector<cv::Point2i> curve;
                cv::approxPolyDP(contour, curve, cv::arcLength(contour, true) * 0.02, true);
                std::vector<std::vector<cv::Point2i>> curves = {curve};
                cv::drawContours(draw, curves, 0, cv::Scalar(255), 2, cv::LINE_AA);
                cv::goodFeaturesToTrack(draw, pts, 4, 0.1, 100);
                std::cout << "找到棋盘" << std::endl;
                // 点序整理
                std::sort(pts.begin(), pts.end(), [](const auto &a, const auto &b)
                          { return a.x < b.x; });
                std::vector<cv::Point2f> left;
                left.emplace_back(pts[0]);
                left.emplace_back(pts[1]);
                std::vector<cv::Point2f> right;
                right.emplace_back(pts[2]);
                right.emplace_back(pts[3]);
                std::sort(left.begin(), left.end(), [](const auto &a, const auto &b)
                          { return a.y < b.y; });
                std::sort(right.begin(), right.end(), [](const auto &a, const auto &b)
                          { return a.y < b.y; });
                pts = std::vector<cv::Point2f>{left[0], right[0], right[1], left[1]};

#ifdef DEBUG
                cv::line(show, pts[0], pts[1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::line(show, pts[1], pts[2], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::line(show, pts[2], pts[3], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::line(show, pts[3], pts[0], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
#endif
                board_->build_board(pts, src_);
                chess_finder.join();
                break;
            }
#ifdef DEBUG
            // cv::imshow("debug", draw);
            cv::imshow("show", show);
#endif
        };
        int low[3] = {110, 75, 0};
        int high[3] = {255, 255, 255};
        cv::Mat src_;
        std::string cam_name_;
        cv::VideoCapture cam_;
        std::shared_ptr<std::thread> camera_stream_;
        std::shared_ptr<std::thread> serial_driver_;
        std::shared_ptr<std::thread> board_finder_;
        std::shared_ptr<edc::Board> board_;
        std::shared_ptr<VisionSerial> serial_;
        std::vector<edc::Chess> chesses_;
    };
}; // namespace edc

int main(int argc, char *argv[])
{
    std::string dev_name = "/dev/video2";
    if (argc > 1)
    {
        dev_name = argv[1];
    }
    auto slover = edc::Slover(dev_name);
    return 0;
}