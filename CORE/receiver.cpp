// 编译命令: g++ -o receiver receiver.cpp
#include <iostream>
#include <fcntl.h>      // open
#include <unistd.h>     // read, close
#include <cstring>
#include <cstdio>

struct CircleData {
    float cx;
    float cy;
    float confidence;
};

int main() {
    const char* pipe_path = "/tmp/vision_pipe";

    // 如果管道不存在则创建 (mkfifo)
    if (access(pipe_path, F_OK) == -1) {
        if (mkfifo(pipe_path, 0666) == -1) {
            perror("mkfifo");
            return -1;
        }
        std::cout << "创建管道: " << pipe_path << std::endl;
    }

    std::cout << "等待视觉模块连接...\n";
    int fd = open(pipe_path, O_RDONLY);
    if (fd == -1) {
        perror("open pipe");
        return -1;
    }
    std::cout << "✅ 已连接，开始接收坐标\n";

    CircleData data;
    while (true) {
        ssize_t n = read(fd, &data, sizeof(data));
        if (n == sizeof(data)) {
            printf("圆心: (%.1f, %.1f) 置信度: %.2f\n", data.cx, data.cy, data.confidence);
            // ====== 在此添加控制逻辑 ======
        } else if (n == 0) {
            // 写端关闭，可能视觉程序退出
            std::cout << "视觉模块已断开连接\n";
            break;
        } else if (n < 0) {
            perror("read");
            break;
        }
    }

    close(fd);
    return 0;
}