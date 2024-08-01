#ifndef SERIAL_TEST_PACKET_H
#define SERIAL_TEST_PACKET_H

#include <cstdint>

#pragma pack(2)
struct VisionMsg
{
    uint16_t head;
    float chess_x;
    float chess_y;
    float dst_x;
    float dst_y;
};

struct RobotMsg
{
    uint16_t head;
    uint8_t task; //task：0x00-对弈 / 0x01 指定黑棋子 / 0x02 指定白棋子 / 0x03-纠错
    uint8_t pos;
};

union VisionPack
{
    struct VisionMsg msg;
    uint8_t bytes[sizeof(VisionMsg)];
};

union RobotPack
{
    struct RobotMsg msg;
    uint8_t bytes[sizeof(RobotMsg)];
};
#pragma pack()

#endif // SERIAL_TEST_PACKET_H