// 编译命令：g++ -o receiver receiver.cpp
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sys/stat.h>

struct VisionData {
    float cx;          // 圆心 x 坐标（像素）
    float cy;          // 圆心 y 坐标
    float confidence;  // 置信度 0~1
};

int main() {
    const char* pipe_path = "/tmp/vision_pipe";

    if (access(pipe_path, F_OK) == -1) {
        mkfifo(pipe_path, 0666);
    }

    std::cout << "等待 Python 连接管道..." << std::endl;
    int fd = open(pipe_path, O_RDONLY);
    if (fd < 0) {
        perror("无法打开管道");
        return 1;
    }
    std::cout << "已连接，开始接收圆心坐标\n" << std::endl;

    VisionData data;
    while (true) {
        ssize_t n = read(fd, &data, sizeof(data));
        if (n == sizeof(data)) {
            printf("圆心: (%.1f, %.1f) 置信度: %.2f\n", data.cx, data.cy, data.confidence);
            // 在这里添加你的控制逻辑
        } else if (n == 0) {
            std::cout << "视觉程序已关闭管道" << std::endl;
            break;
        } else {
            perror("读取错误");
            break;
        }
    }
    close(fd);
    return 0;
}