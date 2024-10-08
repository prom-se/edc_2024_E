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
        -107.2, -106.7, 0};

    const std::array<cv::Point3d, 9> board2pos = {
        cv::Point3d(-32, -32, 0),
        cv::Point3d(0, -32, 0),
        cv::Point3d(32, -32, 0),
        cv::Point3d(-32, 0, 0),
        cv::Point3d(0, 0, 0),
        cv::Point3d(32, 0, 0),
        cv::Point3d(-32, 32, 0),
        cv::Point3d(0, 32, 0),
        cv::Point3d(32, 32, 0),
    };

    const std::array<double, 2> board_dis = {
        45.25483,
        32,
    };

    // TODO：undistort
    const std::array<cv::Point2d, 4> fix_point = {
        cv::Point2d(300, 24),
        cv::Point2d(1238, 12),
        cv::Point2d(1237, 421),
        cv::Point2d(307, 428),
    };

    const double dis2cam = 240; // mm

    const std::array<double, 9> camera_matrix{777.709521106825, 0, 646.907918000159,
                                              0, 776.767897082737, 373.891934779883,
                                              0, 0, 1};
    const std::array<double, 5> dist_coeffs{0.113281809309809, -0.140465800226699, 0, 0, 0};

    class Chess
    {
    public:
        Chess() : index_{9}, color_{edc::BLACK}, pos_{cv::Point2d(0, 0)} {}
        Chess(uint8_t index, edc::ChessColor color, cv::Point2d pos, cv::Point2d pix_pos) : index_{index}, color_{color}, pos_{pos}, pix_pos_{pix_pos} {
                                                                                            };

        uint8_t index()
        {
            return index_;
        };
        edc::ChessColor color()
        {
            return color_;
        };

        cv::Point2d get_pos()
        {
            return pos_;
        };

        cv::Point2d get_pix_pos()
        {
            return pix_pos_;
        };

    private:
        uint8_t index_;
        edc::ChessColor color_;
        cv::Point2d pos_;
        cv::Point2d pix_pos_;
    };

    class Board : public cv::RotatedRect
    {
    public:
        Board() : cam2board_{cv::Point3d()}, net_{Inference("../model/last.onnx", cv::Size(640, 640))},
                  new_chess_map_{cv::Mat(3, 3, CV_8UC1)}, now_chess_map_{cv::Mat(3, 3, CV_8UC1)},
                  black_score_{cv::Mat(3, 3, CV_8SC1)}, white_score_{cv::Mat(3, 3, CV_8SC1)},
                  board_pix_pos_{std::vector<cv::Point2f>(9)}
        {
            cv::Mat(3, 3, CV_64FC1, const_cast<double *>(camera_matrix.data())).copyTo(camera_matrix_);
            cv::Mat(1, 5, CV_64FC1, const_cast<double *>(dist_coeffs.data())).copyTo(dist_coeffs_);
            transform_board_ = cv::Mat(cv::Size(900, 900), CV_8UC3);
        };

        void init_board_pos(cv::RotatedRect &rect)
        {
            std::vector<cv::Point2f> board_pix_pos(9);
            std::vector<cv::Point2f> corner(4);
            std::vector<cv::Point2f> middle(4);
            rect.points(corner);
            auto pts = corner;
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

            corner[0] = left[1];
            corner[1] = left[0];
            corner[2] = right[0];
            corner[3] = right[1];

            middle[0] = (corner[1] + corner[2]) / 2;
            middle[1] = (corner[0] + corner[1]) / 2;
            middle[2] = (corner[2] + corner[3]) / 2;
            middle[3] = (corner[0] + corner[3]) / 2;
            board_pix_pos[0] = corner[1];
            board_pix_pos[1] = middle[0];
            board_pix_pos[2] = corner[2];
            board_pix_pos[3] = middle[1];
            board_pix_pos[4] = rect.center;
            board_pix_pos[5] = middle[2];
            board_pix_pos[6] = corner[0];
            board_pix_pos[7] = middle[3];
            board_pix_pos[8] = corner[3];
            board_pix_pos_ = board_pix_pos;
        }

        void build_board(std::vector<cv::Point2f> &key_points, cv::RotatedRect &rect)
        {
            angle = rect.angle;
            rect.size.width = rect.size.width * 2.0 / 3.0;
            rect.size.height = rect.size.height * 2.0 / 3.0;
            init_board_pos(rect);
            key_points_ = key_points;
            center = (key_points[0] + key_points[1] + key_points[2] + key_points[3]) / 4;
            cv::Mat tvec = cv::Mat_<double>(1, 3);
            cv::Mat rvec = cv::Mat_<double>(1, 3);
            cv::solvePnP(real_size_, key_points, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
            rvec_ = rvec;
            tvec_ = tvec;
            cv::Mat rmat;
            cv::Rodrigues(rvec, rmat);
            Eigen::Matrix3d rotation_matrix;
            cv::cv2eigen(rmat, rotation_matrix);
            auto rpy = rotation_matrix.eulerAngles(2, 1, 0);
            theta_ = rpy[0] / CV_PI * 180.0;
            theta_ = theta_ > 90 ? theta_ - 180 : theta_;
            theta_ = theta_ < -45 ? theta_ + 90 : theta_;
            cam2board_.x = tvec.at<double>(0);
            cam2board_.y = tvec.at<double>(1);
            cam2board_.z = tvec.at<double>(2);
            for (auto &pt : board2pos)
            {
                center2pos_.emplace_back(pt.x * cos(theta_), pt.y * sin(theta_));
            }
        };

        void static detect_chess(edc::Board *self, cv::Mat src)
        {
            if (self->key_points_.empty())
            {
                return;
            }
            self->black_chesses_.clear();
            self->white_chesses_.clear();
            auto detections = self->net_.runInference(src.clone());
            for (auto &detection : detections)
            {
#ifdef DEBUG
                cv::putText(src, detection.className, detection.box.tl(), cv::LINE_AA, 2, detection.color);
                cv::rectangle(src, detection.box, detection.color);
#endif
                cv::Point2d pix_pt(detection.box.x + detection.box.width / 2,
                                   detection.box.y + detection.box.height / 2);

                cv::Point2d chess_pos;
                // chess_pos.x = (dis2cam * ((pix_pt.x - self->camera_matrix_.at<double>(0, 2)) / self->camera_matrix_.at<double>(0, 0))) - cam2org.x;
                // chess_pos.y = (dis2cam * ((pix_pt.y - self->camera_matrix_.at<double>(1, 2)) / self->camera_matrix_.at<double>(1, 1))) - cam2org.y;
                chess_pos = self->remap_position(pix_pt);
                uint8_t index = 9;
                if (cv::pointPolygonTest(self->key_points_, pix_pt, false) >= 0)
                {
                    std::vector<double> dis(9);
                    for (size_t i = 0; i < 9; i++)
                    {
                        dis[i] = cv::norm(pix_pt - self->get_position(i));
                    }
                    index = std::distance(dis.begin(), std::min_element(dis.begin(), dis.end()));
                }
                edc::Chess chess(index, (edc::ChessColor)detection.class_id, chess_pos, pix_pt);
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
        void solve_game(uint8_t &black, uint8_t &white, const uint8_t task)
        {
            update_chesses_map(task);
            get_index(black, white);
        }
        void get_index(uint8_t &black, uint8_t &white)
        {
            black = black_ + 1;
            white = white_ + 1;
        }
        edc::ChessColor get_self_color()
        {
            int count = 0;
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    if (now_chess_map_.at<uint8_t>(i, j))
                    {
                        count++;
                    }
                }
            }
            if (count % 2)
            {
                return WHITE;
            }
            else
            {
                return BLACK;
            }
        }
        uint8_t get_dst_by_color(edc::ChessColor color)
        {
            if (color == edc::BLACK)
            {
                return black_;
            }
            else
            {
                return white_;
            }
        }

        void get_diff(uint8_t &src_index, uint8_t &dst_index)
        {
            auto diff = new_chess_map_ - now_chess_map_;
            for (size_t row = 0; row < 3; row++)
            {
                for (size_t col = 0; col < 3; col++)
                {
                    if (diff.a.at<uint8_t>(row, col) == 1)
                    {
                        src_index = row * 3 + col;
                    }
                    if (diff.a.at<uint8_t>(row, col) == -1)
                    {
                        dst_index = row * 3 + col;
                    }
                }
            }
        };

        uint8_t get_dst()
        {
            int count = 0;
            for (size_t i = 0; i < 3; i++)
            {
                for (size_t j = 0; j < 3; j++)
                {
                    if (now_chess_map_.at<uint8_t>(i, j))
                    {
                        count++;
                    }
                }
            }
            if (count % 2)
            {
                return get_dst_by_color(edc::WHITE);
            }
            else
            {
                return get_dst_by_color(edc::BLACK);
            }
        }

        cv::Point2d remap_position(const cv::Point2d pix_pt)
        {
            // 26.4 11.5
            if (pix_pt.x == 0 && pix_pt.y == 0)
            {
                return pix_pt;
            }
            double k_x = 264 / ((fix_point[1] - fix_point[0] + fix_point[2] - fix_point[3]) / 2).x;
            double k_y = 112.5 / ((fix_point[2] - fix_point[0] + fix_point[3] - fix_point[1]) / 2).y;
            return cv::Point2d((pix_pt.x - (fix_point[0] + fix_point[3]).x / 2) * k_x, (pix_pt.y - (fix_point[0] + fix_point[1]).y / 2) * k_y);
        }

        cv::Point2d get_pnp_position(uint8_t index)
        {
            if (rvec_.empty() || tvec_.empty())
            {
                return cv::Point2d(0, 0);
            }
            double x = (cam2board_ + (board2pos[index] * cos(theta_))).x;
            double y = (cam2board_ + (board2pos[index] * sin(theta_))).y;
            double z = cam2board_.z;
            std::vector<cv::Point3f> pt{cv::Point3f(x, y, z)};
            std::vector<cv::Point2f> img_pt(1);
            cv::projectPoints(pt, rvec_, tvec_, camera_matrix_, dist_coeffs_, img_pt);
            return cv::Point2d(img_pt[0].x, img_pt[0].y);
        };

        // cv::Point2d get_position(uint8_t index)
        // {
        //     double dis = board_dis[index % 2];
        //     std::array<double, 9> thetas = {
        //         135, 90, 45,
        //         180, 0, 0,
        //         -135, -90, -45};
        //     theta_ = thetas[index];
        //     if (index == 4)
        //     {
        //         dis = 0;
        //         theta_ = 0;
        //     }
        //     double x = cam2board_.x + (dis * cos(theta_)) - cam2org.x;
        //     double y = cam2board_.y - (dis * sin(theta_)) - cam2org.y;
        //     return cv::Point2d(x, y);
        // };

        cv::Point2d get_position(uint8_t index)
        {
            if (index > 8)
            {
                return cv::Point2d(0, 0);
            }
            return board_pix_pos_[index];
        };

        cv::Point2d get_src_chess(edc::ChessColor color)
        {
            cv::Point2d points(0, 0);
            if (color == edc::BLACK)
            {
                if (black_chesses_.empty())
                {
                    return points;
                }
                std::vector<edc::Chess> chesses(black_chesses_.begin(), black_chesses_.end());
                if (chesses.empty())
                {
                    return points;
                }
                std::sort(chesses.begin(), chesses.end(), [](auto &a, auto &b)
                          { return a.index() > b.index(); });
                points.x = chesses[0].get_pix_pos().x;
                points.y = chesses[0].get_pix_pos().y;
            }
            else if (color == edc::WHITE)
            {
                if (white_chesses_.empty())
                {
                    return points;
                }
                std::vector<edc::Chess> chesses(white_chesses_.begin(), white_chesses_.end());
                if (chesses.empty())
                {
                    return points;
                }
                std::sort(chesses.begin(), chesses.end(), [](auto &a, auto &b)
                          { return a.index() > b.index(); });
                points.x = chesses[0].get_pix_pos().x;
                points.y = chesses[0].get_pix_pos().y;
            }
            return points;
        }

    private:
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
            uint8_t temp = 0;
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
                            temp++;
                            break;
                        };
                    }
                };
            }
            if (temp == 1)
            {
                score++;
            }
            else
            {
                score--;
            }
            return score;
        }
        void update_chesses_map(uint8_t task = 0x00)
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
            if (task == 0x00 || task == 0x01 || task == 0x02)
            {
                cv::Mat(3, 3, CV_8UC1, const_cast<uint8_t *>(totle.data())).copyTo(now_chess_map_);
            }
            else if (task == 0x03)
            {
                cv::Mat(3, 3, CV_8UC1, const_cast<uint8_t *>(totle.data())).copyTo(new_chess_map_);
            }
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
                        white_score[i] = 15;
                    }
                    if (check_one_step(black, i))
                    {
                        white_score[i] = 10;
                        black_score[i] = 15;
                    }
                    if (i == 4)
                    {
                        black_score[i] = 20;
                        white_score[i] = 20;
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

        std::vector<cv::Point2d> center2pos_;
        std::vector<edc::Chess> black_chesses_;
        std::vector<edc::Chess> white_chesses_;
        cv::Mat new_chess_map_;
        cv::Mat now_chess_map_;
        cv::Mat black_score_;
        cv::Mat white_score_;

        Inference net_;

        uint8_t black_;
        uint8_t white_;
        cv::Mat camera_matrix_;
        cv::Mat dist_coeffs_;
        std::vector<cv::Point2f> board_pix_pos_;
        std::vector<cv::Point2f> key_points_;
        cv::Mat rvec_;
        cv::Mat tvec_;
        cv::Point3d cam2board_;
        double theta_;
        cv::Mat transform_board_;
        const int8_t square_ = 94;
        const std::vector<cv::Point3f> real_size_ = {
            cv::Point3f(-square_ / 2, square_ / 2, 0),
            cv::Point3f(square_ / 2, square_ / 2, 0),
            cv::Point3f(square_ / 2, -square_ / 2, 0),
            cv::Point3f(-square_ / 2, -square_ / 2, 0)};
    };
}; // namespace edc

#endif // BOARD_HPP