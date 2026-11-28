// Optional Linux SDK instrumentation. Delegates every call to the installed SDK.
#include <dexhand_driver/impl/rs485.hpp>
#include <dlfcn.h>
#include <unistd.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace {
struct Stamp { std::uint64_t ros, steady; };
Stamp now() {
    timespec wall{}, steady{};
    clock_gettime(CLOCK_REALTIME, &wall);
    clock_gettime(CLOCK_MONOTONIC, &steady);
    return {std::uint64_t(wall.tv_sec) * 1000000000 + wall.tv_nsec,
            std::uint64_t(steady.tv_sec) * 1000000000 + steady.tv_nsec};
}

class Trace {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> queue;
    bool stopping = false;
    std::thread writer;
    FILE* output = nullptr;
    std::uint64_t written = 0;
    std::atomic<std::uint64_t> dropped{0};

    void run() {
        for (;;) {
            std::deque<std::string> pending;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait_for(lock, std::chrono::milliseconds(100), [&] { return stopping || !queue.empty(); });
                pending.swap(queue);
                if (stopping && pending.empty()) break;
            }
            for (const auto& row : pending) {
                if (std::fwrite(row.data(), 1, row.size(), output) == row.size()) ++written;
                else ++dropped;
            }
            std::fflush(output);
        }
    }
public:
    Trace() {
        const char* path = std::getenv("BOTYARD_IO_TRACE_FILE");
        if (!path || !*path) return;
        output = std::fopen(path, "wx");
        if (!output) { std::perror("Botyard IO trace"); return; }
        std::fprintf(output, "{\"event\":\"io_trace_start\",\"format_version\":1,\"pid\":%d}\n", int(getpid()));
        writer = std::thread([this] { run(); });
    }
    ~Trace() {
        if (!output) return;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_one();
        writer.join();
        std::fprintf(output, "{\"event\":\"io_trace_end\",\"pid\":%d,\"written_events\":%llu,\"dropped_events\":%llu}\n",
                     int(getpid()), (unsigned long long)written, (unsigned long long)dropped.load());
        std::fclose(output);
    }
    void add(std::string row) noexcept {
        if (!output) return;
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (queue.size() >= 4096) { ++dropped; return; }
            queue.push_back(std::move(row));
            changed.notify_one();
        } catch (...) { /* Diagnostic allocation errors must not alter the SDK result. */ }
    }
};

Trace& trace() { static Trace instance; return instance; }
std::atomic<std::uint64_t> sequence{0};
thread_local std::uint64_t request = 0, ack_context = 0;
thread_local unsigned request_command = 256;

template<class Function> Function original(const char* symbol) {
    void* address = dlsym(RTLD_NEXT, symbol);
    if (!address) {
        // pluginlib can load the SDK with RTLD_LOCAL. Resolve its own symbol
        // through the already loaded handle without replacing/loading a driver.
        const char* path = std::getenv("BOTYARD_IO_TRACE_SDK");
        void* handle = dlopen(path ? path : "/usr/local/lib/libdexhand_driver.so", RTLD_NOW | RTLD_NOLOAD);
        if (handle) { address = dlsym(handle, symbol); dlclose(handle); }
    }
    if (!address) { std::fprintf(stderr, "Botyard IO trace: missing SDK symbol %s\n", symbol); std::abort(); }
    return reinterpret_cast<Function>(address);
}

void emit(const char* kind, unsigned cmd, Stamp begin, Stamp end, bool success,
          const std::vector<std::uint8_t>* data = nullptr) noexcept {
    try {
        std::ostringstream row;
        row << "{\"event\":\"" << kind << "\",\"pid\":" << getpid()
            << ",\"request_id\":" << (request_command == cmd ? request : 0)
            << ",\"ack_context_id\":" << ack_context << ",\"cmd_id\":" << cmd
            << ",\"begin_ros_ns\":" << begin.ros << ",\"end_ros_ns\":" << end.ros
            << ",\"begin_steady_ns\":" << begin.steady << ",\"end_steady_ns\":" << end.steady
            << ",\"success\":" << (success ? "true" : "false");
        if (data) {
            static const char hex[] = "0123456789abcdef";
            row << ",\"data_size\":" << data->size() << ",\"data_hex\":\"";
            // Requests include the encoded angle/speed/effort/tactile fields.
            // Responses need only a prefix to distinguish a received NACK.
            const auto limit = std::string(kind) == "serial_send" ? 4096u : 8u;
            for (std::size_t i = 0; i < data->size() && i < limit; ++i)
                row << hex[(*data)[i] >> 4] << hex[(*data)[i] & 15];
            row << '"';
        }
        row << "}\n";
        trace().add(row.str());
    } catch (...) { /* Preserve SDK behavior even when diagnostics cannot allocate. */ }
}
}  // namespace

namespace dexhand_driver {
bool SerialDriver::sendCommand(std::uint8_t cmd, const std::vector<std::uint8_t>& data, std::uint8_t hand) {
    using Function = bool (*)(SerialDriver*, std::uint8_t, const std::vector<std::uint8_t>&, std::uint8_t);
    static auto call = original<Function>("_ZN14dexhand_driver12SerialDriver11sendCommandEhRKSt6vectorIhSaIhEEh");
    trace();  // Initialize diagnostics before timing the actual SDK call.
    request = ++sequence; request_command = cmd;
    const auto begin = now();
    const bool result = call(this, cmd, data, hand);
    emit("serial_send", cmd, begin, now(), result, &data);
    return result;
}

bool SerialDriver::waitResponse(std::uint8_t cmd, std::vector<std::uint8_t>& response) {
    using Function = bool (*)(SerialDriver*, std::uint8_t, std::vector<std::uint8_t>&);
    static auto call = original<Function>("_ZN14dexhand_driver12SerialDriver12waitResponseEhRSt6vectorIhSaIhEE");
    const auto begin = now();
    const bool result = call(this, cmd, response);
    emit("serial_response", cmd, begin, now(), result, &response);
    return result;
}

bool SerialDriver::sendCommandAndCheckAck(std::uint8_t cmd, const std::vector<std::uint8_t>& data, std::uint8_t hand) {
    using Function = bool (*)(SerialDriver*, std::uint8_t, const std::vector<std::uint8_t>&, std::uint8_t);
    static auto call = original<Function>("_ZN14dexhand_driver12SerialDriver22sendCommandAndCheckAckEhRKSt6vectorIhSaIhEEh");
    trace();
    struct Restore { std::uint64_t old; ~Restore() { ack_context = old; } } restore{ack_context};
    ack_context = ++sequence;
    const auto begin = now();
    const bool result = call(this, cmd, data, hand);
    emit("serial_ack", cmd, begin, now(), result);
    return result;
}
}  // namespace dexhand_driver
