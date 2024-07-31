#ifndef BOARD_HPP
#define BOARD_HPP

#include <map>
#include <array>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/dnn.hpp>
#define DEBUG

namespace edc
{
    enum ChessLocation
    {
        CENTER,
        EDGE,
        CORNER,
    };
    enum ChessColor
    {
        BLACK,
        WHITE,
    };
    static std::unordered_map<uint8_t, edc::ChessLocation> location_map =
        {
            {1, CORNER},
            {2, EDGE},
            {3, CORNER},
            {4, EDGE},
            {5, CENTER},
            {6, EDGE},
            {7, CORNER},
            {8, EDGE},
            {9, CORNER},
    };

    const cv::Point3d cam2org = {
        0, 0, 0};

    const std::array<cv::Point3d, 9> board2pos = {
        cv::Point3d(-30, -30, 0),
        cv::Point3d(0, -30, 0),
        cv::Point3d(30, -30, 0),
        cv::Point3d(-30, 0, 0),
        cv::Point3d(0, 0, 0),
        cv::Point3d(30, 0, 0),
        cv::Point3d(-30, 30, 0),
        cv::Point3d(0, 30, 0),
        cv::Point3d(30, 30, 0),
    };

    const double dis2cam = 100; // mm

    const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 773.087485, 0.0, 649.0052766,
                                   0.0, 772.4994227, 324.1908488,
                                   0.0, 0.0, 1.0);

    const cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << 0.09430021, -0.073412315, 0, 0, 0);

    const std::vector<cv::Point2f> transform_points =
        {
            cv::Point2f(0, 900), cv::Point2f(0, 0), cv::Point2f(900, 0), cv::Point2f(900, 900)};

    class Chess
    {
    public:
        uint8_t index_;

        edc::ChessColor color()
        {
            return color_;
        };
        edc::ChessLocation location()
        {
            location_ = location_map[index_];
            return location_;
        }

    private:
        edc::ChessColor color_;
        edc::ChessLocation location_;
    };

    class Board : public cv::RotatedRect
    {
    public:
        Board() : cam2board_{cv::Point3d()}
        {
            net_ = cv::dnn::readNetFromONNX("model/chess.onnx");
            transform_board_ = cv::Mat(cv::Size(900, 900), CV_8UC3);
        };

        void build_board(std::vector<cv::Point2f> key_points, cv::Mat src)
        {
            center_ = (key_points[0] + key_points[1] + key_points[2] + key_points[3]) / 4;
            std::vector<double> rvec;
            std::vector<double> tvec;
            cv::solvePnP(real_size_, key_points, camera_matrix, dist_coeffs, rvec, tvec, cv::SOLVEPNP_IPPE_SQUARE);
            cv::Mat rmat;
            cv::Rodrigues(rvec, rmat);
            Eigen::Matrix3d rotation_matrix;
            cv::cv2eigen(rmat, rotation_matrix);
            auto rpy = rotation_matrix.eulerAngles(2, 1, 0);
            theta_ = rpy[0];
            cam2board_.x = tvec[0];
            cam2board_.y = tvec[1];
            cam2board_.z = tvec[2];
            auto transform_mat_ = cv::getPerspectiveTransform(key_points, transform_points);
            cv::warpPerspective(src, transform_board_, transform_mat_, transform_board_.size());
#ifdef DEBUG
            cv::imshow("debug", transform_board_);
            std::cout << cam2board_.x << '/' << cam2board_.y << '/' << cam2board_.z << '/' << theta_ << std::endl;
#endif
        };

        cv::Point2d get_position(const uint8_t index)
        {
            double x = (cam2board_ + board2pos[index] - cam2org).x;
            double y = (cam2board_ + board2pos[index] - cam2org).x;
            return cv::Point2d(x, y);
#ifdef DEBUG
            std::cout << "index:" << index << x << '/' << y << std::endl;
#endif
        };

        void detect_chess(cv::Mat src)
        {
            auto blob = cv::dnn::blobFromImage(src, 1.0, cv::Size(640, 640));
            net_.setInput(blob);
            cv::Mat output = net_.forward();
#ifdef DEBUG
#endif
        }

    private:
        cv::dnn::Net net_;
        cv::Point3d cam2board_;
        double theta_;
        cv::Point2f center_;
        cv::Mat transform_board_;
        const int8_t square_ = 90;
        const std::vector<cv::Point3f> real_size_ = {
            cv::Point3f(-square_ / 2, square_ / 2, 0),
            cv::Point3f(square_ / 2, square_ / 2, 0),
            cv::Point3f(square_ / 2, -square_ / 2, 0),
            cv::Point3f(-square_ / 2, -square_ / 2, 0)};
    };
}; // namespace edc

#endif // BOARD_HPP