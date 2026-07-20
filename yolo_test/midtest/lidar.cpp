#include <mavsdk/mavsdk.hpp>
#include <mavsdk/plugins/telemetry/telemetry.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

using namespace mavsdk;
using namespace std::this_thread;
using namespace std::chrono;

#define PIPE_PATH "/tmp/lidar_height_pipe"
#define PIPE_PATH_FALLBACK "/home/hy/lidar_height_pipe"

bool create_pipe(const char* path) {
    unlink(path);
    if (mkfifo(path, 0666) == 0) {
        chmod(path, 0666);
        return true;
    }
    return false;
}

int main() {
    // 自动创建管道
    const char* pipe_path = PIPE_PATH;
    if (!create_pipe(pipe_path)) {
        std::cerr << "在 " << PIPE_PATH << " 创建管道失败: " << strerror(errno) << std::endl;
        pipe_path = PIPE_PATH_FALLBACK;
        if (!create_pipe(pipe_path)) {
            std::cerr << "备用路径也失败: " << strerror(errno) << std::endl;
            return 1;
        }
    }
    std::cout << "管道已创建: " << pipe_path << std::endl;

    // 打开管道（读写模式）
    int pipe_fd = open(pipe_path, O_RDWR);
    if (pipe_fd < 0) {
        std::cerr << "打开管道失败: " << strerror(errno) << std::endl;
        return 1;
    }
    std::cout << "管道已成功打开（读写模式）" << std::endl;

    // 飞控连接
    Mavsdk mavsdk{Mavsdk::Configuration{ComponentType::CompanionComputer}};
    auto conn_result = mavsdk.add_any_connection("serial:///dev/ttyTHS1:57600");
    if (conn_result != ConnectionResult::Success) {
        std::cerr << "串口连接失败: " << conn_result << std::endl;
        return 1;
    }

    std::cout << "等待飞控连接..." << std::endl;
    auto start = steady_clock::now();
    while (mavsdk.systems().empty()) {
        if (duration_cast<seconds>(steady_clock::now() - start).count() > 20) {
            std::cerr << "飞控连接超时" << std::endl;
            return 1;
        }
        sleep_for(100ms);
    }

    auto system = mavsdk.systems().at(0);
    Telemetry telemetry{system};
    std::cout << "飞控已连接，订阅激光雷达..." << std::endl;

    // 订阅距离传感器
    telemetry.subscribe_distance_sensor([&](Telemetry::DistanceSensor distance_sensor) {
        float h = distance_sensor.current_distance_m;
        if (h > 0.1f && h < 20.0f) {
            // 写入管道（Python 会读取）
            ssize_t written = write(pipe_fd, &h, sizeof(float));
            if (written == sizeof(float)) {
                // 打印到终端（便于调试）
                std::cout << "激光雷达高度: " << h << " m (已写入管道)" << std::endl;
            } else {
                // 管道可能无读端，忽略
            }
        }
    });

    std::cout << "激光雷达已订阅，等待数据..." << std::endl;

    while (true) {
        sleep_for(1s);
    }

    close(pipe_fd);
    return 0;
}