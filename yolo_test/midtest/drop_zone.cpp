#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sys/stat.h>
#include <vector>

struct BucketInfo {
    unsigned char id;   // 桶编号 1,2,3
    float cx;
    float cy;
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
    std::cout << "已连接，开始接收数据（桶ID + 坐标）" << std::endl;

    while (true) {
        unsigned char count;
        ssize_t n = read(fd, &count, 1);
        if (n == 0) {
            std::cout << "Python 端已关闭管道" << std::endl;
            break;
        } else if (n < 0) {
            perror("读取错误");
            break;
        }

        if (count == 0) {
            std::cout << "None (未检测到桶)" << std::endl;
        } else {
            std::cout << "检测到 " << (int)count << " 个桶:" << std::endl;
            for (int i = 0; i < count; i++) {
                BucketInfo bucket;
                // 读取1字节ID
                n = read(fd, &bucket.id, 1);
                if (n != 1) break;
                // 读取cx, cy (2个float)
                n = read(fd, &bucket.cx, sizeof(float));
                if (n != sizeof(float)) break;
                n = read(fd, &bucket.cy, sizeof(float));
                if (n != sizeof(float)) break;

                std::cout << "  桶" << (int)bucket.id << ": (" << bucket.cx << ", " << bucket.cy << ")" << std::endl;
            }
        }
        std::cout << std::endl;
    }

    close(fd);
    return 0;
}