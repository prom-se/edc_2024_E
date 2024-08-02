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
        Slover(std::string cam_name = "/dev/video2") : show_{cv::Mat(cv::Size(1280, 720), CV_8UC3)}, src_{cv::Mat(cv::Size(1280, 720), CV_8UC3)}
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
            cam_.set(cv::CAP_PROP_EXPOSURE, exp);
            cam_.set(cv::CAP_PROP_AUTO_WB, 1);
            // cam_.set(cv::CAP_PROP_WB_TEMPERATURE,4850);
#ifdef DEBUG
            // cv::namedWindow("src");
            cv::namedWindow("debug");
            cv::namedWindow("show");
            cv::createTrackbar("exp", "debug", &exp, 1000);
            cv::createTrackbar("l_low", "debug", low, 255);
            cv::createTrackbar("a_low", "debug", low + 1, 255);
            cv::createTrackbar("b_low", "debug", low + 2, 255);
            cv::createTrackbar("l_high", "debug", high, 255);
            cv::createTrackbar("a_high", "debug", high + 1, 255);
            cv::createTrackbar("b_high", "debug", high + 2, 255);
#endif
            while (true)
            {
                cam_.set(cv::CAP_PROP_EXPOSURE, exp);
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
                    serial_->get_robot(robot_);
                    uint8_t chesses[9];
                    float angle;
                    vision_.head = 0xA5;
                    uint8_t black, white;
                    cv::Point2d fix{-3,-1};
                    cv::Point2d point = board_->remap_position(board_->get_src_chess(board_->get_self_color()));
                    point += fix;
                    vision_.chess_x = point.x > 0 ? point.x : vision_.chess_x;
                    vision_.chess_y = point.y > 0 ? point.y : vision_.chess_y;
                    if (robot_.task == 0x00)
                    {
                        vision_.dst_x = board_->get_position(board_->get_dst()).x;
                        vision_.dst_y = board_->get_position(board_->get_dst()).y;
                    }
                    else if (robot_.task == 0x01)
                    {
                        vision_.dst_x = board_->get_position(board_->get_dst_by_color(edc::BLACK) - 1).x;
                        vision_.dst_y = board_->get_position(board_->get_dst_by_color(edc::BLACK) - 1).y;
                    }
                    else if (robot_.task == 0x02)
                    {
                        vision_.dst_x = board_->get_position(board_->get_dst_by_color(edc::WHITE) - 1).x;
                        vision_.dst_y = board_->get_position(board_->get_dst_by_color(edc::WHITE) - 1).y;
                    }
                    else if (robot_.task == 0x03)
                    {
                        cv::Point2d src(0, 0);
                        cv::Point2d dst(0, 0);
                        uint8_t src_index = 0;
                        uint8_t dst_index = 0;
                        board_->get_diff(src_index, dst_index);
                        src = board_->get_position(src_index);
                        dst = board_->get_position(dst_index);
                        src += fix;
                        vision_.chess_x = src.x > 0 ? src.x : vision_.chess_x;
                        vision_.chess_y = src.y > 0 ? src.y : vision_.chess_y;
                        vision_.dst_x = dst.x;
                        vision_.dst_y = dst.y;
                    }
                    vision_.dst_x += fix.x - 10;
                    vision_.dst_y += fix.y - 5;
                    if (vision_.chess_x < 0)
                    {
                        vision_.chess_x = 0;
                    }
                    if (vision_.chess_y < 0)
                    {
                        vision_.chess_y = 0;
                    }
                    if (vision_.dst_x < 0)
                    {
                        vision_.dst_x = 0;
                    }
                    if (vision_.dst_y < 0)
                    {
                        vision_.dst_y = 0;
                    }
                    std::cout << board_->get_self_color() << '/';
                    std::cout << "src:" << vision_.chess_x << '/' << vision_.chess_y << '/';
                    std::cout << "dst" << vision_.dst_x << '/' << vision_.dst_y << std::endl;
                    serial_->update_vision(vision_);
                }
                catch (std::exception &ex)
                {
                }
            }
        };

        void find_board()
        {
            // 预处理
            cv::Mat undistort;
            cv::undistort(src_, undistort, cv::Mat(3, 3, CV_64FC1, const_cast<double *>(camera_matrix.data())), cv::Mat(1, 5, CV_64FC1, const_cast<double *>(dist_coeffs.data())));
            show_ = undistort.clone();
            std::thread chess_finder(std::bind(&edc::Board::detect_chess, board_.get(), show_));
            cv::Mat lab(cv::Mat::zeros(720, 1280, CV_8UC3));
            cv::Mat bin(cv::Mat::zeros(720, 1280, CV_8UC1));
            cv::Mat draw(cv::Mat::zeros(720, 1280, CV_8UC1));
            cv::cvtColor(src_.clone(), lab, cv::COLOR_BGR2Lab);
            cv::inRange(lab, cv::Scalar(low[0], low[1], low[2]), cv::Scalar(high[0], high[1], high[2]), bin);
            cv::threshold(bin.clone(), bin, 0, 255, cv::THRESH_BINARY_INV);
            cv::morphologyEx(bin.clone(), bin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
            std::vector<std::vector<cv::Point2i>> contours;
            cv::findContoursLinkRuns(bin, contours);
            std::array<cv::Point2f, 4> pts;
            for (auto &contour : contours)
            {
                // 面积筛选
                auto rect = cv::minAreaRect(contour);
                if (abs(rect.size.width / rect.size.height - 1) > 0.15)
                {
                    continue;
                }
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
                rect.points(pts);

#ifdef DEBUG
                cv::line(show_, pts[0], pts[1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::line(show_, pts[1], pts[2], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::line(show_, pts[2], pts[3], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::line(show_, pts[3], pts[0], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                // cv::putText(show_, "0", pts[0], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                // cv::putText(show_, "1", pts[1], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                // cv::putText(show_, "2", pts[2], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                // cv::putText(show_, "3", pts[3], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
#endif
                board_->build_board(pts);
                break;
            }
            if (chess_finder.joinable())
            {
                chess_finder.join();
            }
            uint8_t black = 9;
            uint8_t white = 9;
            board_->solve_game(black, white, robot_.task);
#ifdef DEBUG
            cv::putText(show_, cv::format("BLACK:%d", black), cv::Point2i(50, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::putText(show_, cv::format("WHITE:%d", white), cv::Point2i(50, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::imshow("debug", bin);
            cv::imshow("show", show_);
#endif
        };
        int exp = 300;
        int low[3] = {35, 43, 0};
        int high[3] = {255, 185, 183};
        cv::Mat show_;
        cv::Mat src_;
        std::string cam_name_;
        cv::VideoCapture cam_;
        VisionMsg vision_;
        RobotMsg robot_;
        std::shared_ptr<std::thread> camera_stream_;
        std::shared_ptr<std::thread> serial_driver_;
        std::shared_ptr<std::thread> board_finder_;
        std::shared_ptr<edc::Board> board_;
        std::shared_ptr<VisionSerial> serial_;
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