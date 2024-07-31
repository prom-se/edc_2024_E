#ifndef BOARD_HPP
#define BOARD_HPP

#include <map>
#include <array>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/dnn.hpp>
#include "inference.h"

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
    static std::array<std::array<uint8_t, 3>, 8> win_map =
        {
            std::array<uint8_t, 3>{0, 1, 2},
            std::array<uint8_t, 3>{3, 4, 5},
            std::array<uint8_t, 3>{6, 7, 8},
            std::array<uint8_t, 3>{0, 3, 6},
            std::array<uint8_t, 3>{1, 4, 7},
            std::array<uint8_t, 3>{2, 5, 8},
            std::array<uint8_t, 3>{0, 4, 8},
            std::array<uint8_t, 3>{2, 4, 6},
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

    const double dis2cam = 240; // mm

    const std::array<double, 9> camera_matrix{777.709521106825, 0, 646.907918000159,
                                              0, 776.767897082737, 373.891934779883,
                                              0, 0, 1};
    const std::array<double, 5> dist_coeffs{0.113281809309809, -0.140465800226699, 0, 0, 0};

    const std::vector<cv::Point2f> transform_points =
        {
            cv::Point2f(0, 900), cv::Point2f(0, 0), cv::Point2f(900, 0), cv::Point2f(900, 900)};

    class Chess
    {
    public:
        Chess(uint8_t index, edc::ChessColor color, cv::Point2d pos) : index_{index}, color_{color}, pos_{pos} {
                                                                       };

        uint8_t index()
        {
            return index_;
        };
        edc::ChessColor color()
        {
            return color_;
        };

    private:
        uint8_t index_;
        edc::ChessColor color_;
        cv::Point2d pos_;
    };

    class Board : public cv::RotatedRect
    {
    public:
        Board() : cam2board_{cv::Point3d()}, net_{Inference("../model/chess.onnx", cv::Size(640, 640))},
                  last_chess_map_{cv::Mat(3, 3, CV_8UC1)}, now_chess_map_{cv::Mat(3, 3, CV_8UC1)},
                  black_score_{cv::Mat(3, 3, CV_8SC1)}, white_score_{cv::Mat(3, 3, CV_8SC1)}
        {
            cv::Mat(3, 3, CV_64FC1, const_cast<double *>(camera_matrix.data())).copyTo(camera_matrix_);
            cv::Mat(1, 5, CV_64FC1, const_cast<double *>(dist_coeffs.data())).copyTo(dist_coeffs_);
            transform_board_ = cv::Mat(cv::Size(900, 900), CV_8UC3);
        };

        void build_board(std::vector<cv::Point2f> key_points, cv::Mat src)
        {
            key_points_ = key_points;
            center_ = (key_points[0] + key_points[1] + key_points[2] + key_points[3]) / 4;
            cv::Mat tvec = cv::Mat_<double>(1, 3);
            cv::Mat rvec = cv::Mat_<double>(1, 3);
            cv::solvePnP(real_size_, key_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
            cv::Mat rmat;
            cv::Rodrigues(rvec, rmat);
            Eigen::Matrix3d rotation_matrix;
            cv::cv2eigen(rmat, rotation_matrix);
            auto rpy = rotation_matrix.eulerAngles(2, 1, 0);
            theta_ = rpy[0] / CV_PI * 180.0;
            theta_ = theta_ > 90 ? theta_ - 180 : theta_;
            cam2board_.x = tvec.at<double>(0);
            cam2board_.y = tvec.at<double>(1);
            cam2board_.z = tvec.at<double>(2);
            auto transform_mat_ = cv::getPerspectiveTransform(key_points, transform_points);
            cv::warpPerspective(src, transform_board_, transform_mat_, transform_board_.size());
#ifdef DEBUG
            // cv::imshow("debug", transform_board_);
            std::cout << cam2board_.x << '/' << cam2board_.y << '/' << cam2board_.z << '/' << theta_ << std::endl;
#endif
        };

        void static detect_chess(edc::Board *self, cv::Mat &src)
        {
            if (self->key_points_.empty())
            {
                return;
            }
            self->black_chesses_.clear();
            self->white_chesses_.clear();
            auto detections = self->net_.runInference(src);
            for (auto &detection : detections)
            {
#ifdef DEBUG
                // cv::putText(src, detection.className, detection.box.tl(), cv::LINE_AA, 1, detection.color);
                // cv::rectangle(src, detection.box, detection.color);
#endif
                cv::Point2f pix_pt(detection.box.x + detection.box.width / 2,
                                   detection.box.y + detection.box.height / 2);

                cv::Point2d chess_pos;
                chess_pos.x = (dis2cam * ((pix_pt.x - self->camera_matrix_.at<double>(0, 2)) / self->camera_matrix_.at<double>(0, 0))) - cam2org.x;
                chess_pos.y = (dis2cam * ((pix_pt.y - self->camera_matrix_.at<double>(1, 2)) / self->camera_matrix_.at<double>(1, 1))) - cam2org.y;
                uint8_t index = 9;
                if (cv::pointPolygonTest(self->key_points_, pix_pt, false) >= 0)
                {
                    std::vector<double> dis(9);
                    for (size_t i = 0; i < 9; i++)
                    {
                        dis[i] = cv::norm(chess_pos - self->get_position(i));
                    }
                    index = std::distance(dis.begin(), std::min_element(dis.begin(), dis.end()));
                }
                edc::Chess chess(index, (edc::ChessColor)detection.class_id, chess_pos);
                if (chess.color() == edc::BLACK)
                {
                    self->black_chesses_.emplace_back(chess);
                }
                else if (chess.color() == edc::WHITE)
                {
                    self->white_chesses_.emplace_back(chess);
                }
            }
        }
        void solve_game(uint8_t &black, uint8_t &white)
        {
            update_chesses_map();
            black = black_ + 1;
            white = white_ + 1;
        }

    private:
        cv::Point2d get_position(uint8_t index)
        {
            double x = (cam2board_ + board2pos[index] - cam2org).x;
            double y = (cam2board_ + board2pos[index] - cam2org).y;
            return cv::Point2d(x, y);
#ifdef DEBUG
            std::cout << "index:" << index << x << '/' << y << std::endl;
#endif
        };
        bool check_one_step(std::vector<uint8_t> &index, uint8_t fill)
        {
            for (size_t i = 0; i < index.size(); i++)
            {
                for (size_t j = i + 1; j < index.size(); j++)
                {
                    std::array<uint8_t, 3> test{index[i], index[j], fill};
                    std::sort(test.begin(), test.end(), [](const auto &a, const auto &b)
                              { return a < b; });
                    if (std::find(win_map.begin(), win_map.end(), test) != win_map.end())
                    {
                        return true;
                    }
                }
            }
            return false;
        };

        uint8_t check_all_step(std::vector<uint8_t> &foe_index, uint8_t check_index)
        {
            uint8_t score = 0;
            for (size_t i = 0; i < win_map.size(); i++)
            {
                auto check = win_map[i];
                if (std::count(check.begin(), check.end(), check_index))
                {
                    score++;
                    for (size_t j = 0; j < foe_index.size(); j++)
                    {
                        if (std::count(check.begin(), check.end(), foe_index[j]))
                        {
                            score--;
                            break;
                        };
                    }
                };
            }
            return score;
        }
        void update_chesses_map()
        {
            std::vector<uint8_t> black_index;
            std::vector<uint8_t> black = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            std::vector<uint8_t> white_index;
            std::vector<uint8_t> white = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            std::vector<uint8_t> totle = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (auto &black_chesse : black_chesses_)
            {
                if (black_chesse.index() == 9)
                {
                    continue;
                }
                black_index.emplace_back(black_chesse.index());
                black[black_chesse.index()] = 1;
            }
            for (auto &white_chesse : white_chesses_)
            {
                if (white_chesse.index() == 9)
                {
                    continue;
                }
                white_index.emplace_back(white_chesse.index());
                white[white_chesse.index()] = 1;
            }
            std::sort(black_index.begin(), black_index.end(), [](const auto &a, const auto &b)
                      { return a < b; });
            std::sort(white_index.begin(), white_index.end(), [](const auto &a, const auto &b)
                      { return a < b; });
            for (size_t i = 0; i < 9; i++)
            {
                totle[i] = black[i] | white[i];
            }
            now_chess_map_.copyTo(last_chess_map_);
            cv::Mat(3, 3, CV_8UC1, const_cast<uint8_t *>(totle.data())).copyTo(now_chess_map_);
            update_score(black_index, white_index);
        };

        void update_score(std::vector<uint8_t> &black, std::vector<uint8_t> &white)
        {
            std::array<int8_t, 9> black_score = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            std::array<int8_t, 9> white_score = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (uint8_t i = 0; i < 9; i++)
            {
                if (std::count(black.begin(), black.end(), i) || std::count(white.begin(), white.end(), i))
                {
                    black_score[i] = -1;
                    white_score[i] = -1;
                }
                else
                {
                    black_score[i] = check_all_step(white, i);
                    white_score[i] = check_all_step(black, i);
                    if (check_one_step(white, i))
                    {
                        black_score[i] = 10;
                    }
                    if (check_one_step(black, i))
                    {
                        white_score[i] = 10;
                    }
                    if (i == 4)
                    {
                        black_score[i] = 10;
                        white_score[i] = 10;
                    }
                }
            }
            int black_index = 9;
            int white_index = 9;
            cv::minMaxIdx(black_score, nullptr, nullptr, nullptr, &black_index);
            cv::minMaxIdx(white_score, nullptr, nullptr, nullptr, &white_index);
            black_ = black_index;
            white_ = white_index;
        }

        std::vector<edc::Chess> black_chesses_;
        std::vector<edc::Chess> white_chesses_;
        cv::Mat last_chess_map_;
        cv::Mat now_chess_map_;
        cv::Mat black_score_;
        cv::Mat white_score_;

        Inference net_;

        uint8_t black_;
        uint8_t white_;
        cv::Mat camera_matrix_;
        cv::Mat dist_coeffs_;
        std::vector<cv::Point2f> key_points_;
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