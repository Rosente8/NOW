#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <sys/stat.h>

int main() {
    const char* pipe_path = "/tmp/h_pipe";

    if (access(pipe_path, F_OK) == -1) {
        mkfifo(pipe_path, 0666);
    }

    std::cout << "等待 Python 连接 H 管道..." << std::endl;
    int fd = open(pipe_path, O_RDONLY);
    if (fd < 0) {
        perror("无法打开 H 管道");
        return 1;
    }
    std::cout << "已连接 H 管道，开始接收数据" << std::endl;

    // 使用文件描述符转换为 FILE* 以便使用 getline
    FILE* pipe_file = fdopen(fd, "r");
    if (!pipe_file) {
        perror("fdopen 失败");
        close(fd);
        return 1;
    }

    char* line = nullptr;
    size_t len = 0;
    while (getline(&line, &len, pipe_file) != -1) {
        // 去除末尾换行符
        std::string msg(line);
        if (!msg.empty() && msg.back() == '\n') {
            msg.pop_back();
        }
        if (msg == "None") {
            std::cout << "未检测到 H" << std::endl;
        } else {
            // 解析坐标 "cx,cy"
            size_t comma = msg.find(',');
            if (comma != std::string::npos) {
                float cx = std::stof(msg.substr(0, comma));
                float cy = std::stof(msg.substr(comma + 1));
                std::cout << "H 中心: (" << cx << ", " << cy << ")" << std::endl;
            } else {
                std::cout << "收到无效消息: " << msg << std::endl;
            }
        }
    }

    free(line);
    fclose(pipe_file);
    close(fd);
    return 0;
}