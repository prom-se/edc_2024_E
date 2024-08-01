#ifndef VISION_serial_IAL_HPP
#define VISION_serial_IAL_HPP

#include <thread>
#include <vector>
#include <memory>
#include <unistd.h>
#include <CSerialPort/SerialPort.h>
#include <string.h>
#include <iostream>
#include "packet.hpp"

using namespace itas109;

/*
@brief 视觉串口通信类,实例化后使用visionUpdate传入自瞄信息;使用robotUpdate接收机器人信息。
*/
class VisionSerial
{
public:
    /*
    @brief 构造函数，设置串口设备名称和波特率
    @param devName_ 串口设备名称
    @param baudRate_ 波特率
    */
    explicit VisionSerial(const char *dev_name, const int baud_rate)
        : isOk{false}, dev_name_{dev_name}, baud_rate_{baud_rate}, serial_{new CSerialPort(dev_name_)}
    {
        watchdog_thread_ = std::thread(&VisionSerial::WatchDogThreadFun, this);
        send_thread_ = std::thread(&VisionSerial::SendThreadFun, this);
        recive_thread_ = std::thread(&VisionSerial::ReceiveThreadFun, this);
    };
    /*
    @brief 析构函数,用来结束线程和关闭串口。
    */
    ~VisionSerial()
    {
        if (watchdog_thread_.joinable())
        {
            watchdog_thread_.join();
        }
        if (send_thread_.joinable())
        {
            send_thread_.join();
        }
        if (recive_thread_.joinable())
        {
            recive_thread_.join();
        }
        isOk = false;
        if (serial_->isOpen())
        {
            serial_->close();
        }
    };

    /*
    @brief 更新发送的视觉信息
    @param vision 视觉信息结构体
    */
    void update_vision(VisionMsg &vision)
    {
        vision.head = 0xA5;
        vision_pack_.msg = vision;
    };

    void get_robot(RobotMsg &robot){
        robot = robot_pack_.msg;
    };

    bool isOk;

private:
    /*
    @brief 发送线程函数
    */
    void SendThreadFun()
    {
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (isOk)
            {
                serial_->writeData(vision_pack_.bytes, sizeof(VisionPack));
                isOk = serial_->isOpen();
            }
        }
    };

    /*
    @brief 接收线程函数
    */
    void ReceiveThreadFun()
    {
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            std::vector<uint8_t> head(2);
            std::vector<uint8_t> bytes(sizeof(RobotPack) - 2);
            serial_->readData(head.data(), 2);
            if (head[0] == 0xA5 && head[1] == 0x00)
            {
                serial_->readData(bytes.data(), sizeof(RobotPack) - 2);
                bytes.reserve(sizeof(RobotPack));
                bytes.insert(bytes.begin(), head[1]);
                bytes.insert(bytes.begin(), head[0]);
                std::memcpy(robot_pack_.bytes, bytes.data(), sizeof(RobotPack));
            }
            isOk = serial_->isOpen();
        }
    };

    /*
    @brief 看门狗线程函数
    */
    void WatchDogThreadFun()
    {
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            if (isOk)
            {
                return;
            }
            else
            {
                std::cout << "串口重启..." << std::endl;
                serial_->init(dev_name_, baud_rate_);
                isOk = serial_->open();
            }
        }
    };

    const char *dev_name_;
    const int baud_rate_;
    std::unique_ptr<CSerialPort> serial_;
    std::thread send_thread_;
    std::thread recive_thread_;
    std::thread watchdog_thread_;
    VisionPack vision_pack_;
    RobotPack robot_pack_;
};

#endif // VISION_serial_IAL_HPP