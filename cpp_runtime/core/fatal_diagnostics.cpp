#include "mfq/fatal_diagnostics.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <exception>
#include <initializer_list>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#include <crtdbg.h>
#else
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#endif

namespace mfq {
namespace {
std::atomic_flag reporting = ATOMIC_FLAG_INIT;

void write_error(const char* text) noexcept {
    if (!text) return;
    const auto length = std::strlen(text);
#ifdef _WIN32
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), text, static_cast<DWORD>(length), &written, nullptr);
#else
    const char* cursor = text;
    auto remaining = length;
    while (remaining) {
        const auto written = ::write(STDERR_FILENO, cursor, remaining);
        if (written <= 0) break;
        cursor += written;
        remaining -= static_cast<std::size_t>(written);
    }
#endif
}

#ifdef _WIN32
bool symbols_ready = false;
HANDLE symbol_process = nullptr;

void print_address(void* address) noexcept {
    alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME]{};
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(storage);
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;
    DWORD64 displacement = 0;
    const auto pc = reinterpret_cast<DWORD64>(address);
    char line[2048];
    if (symbols_ready && SymFromAddr(symbol_process, pc, &displacement, symbol)) {
        std::snprintf(line, sizeof(line), "  %p %s+0x%llx", address, symbol->Name,
                      static_cast<unsigned long long>(displacement));
        write_error(line);
        IMAGEHLP_LINE64 source{};
        source.SizeOfStruct = sizeof(source);
        DWORD offset = 0;
        if (SymGetLineFromAddr64(symbol_process, pc, &offset, &source)) {
            std::snprintf(line, sizeof(line), " (%s:%lu)", source.FileName, source.LineNumber);
            write_error(line);
        }
    } else {
        MEMORY_BASIC_INFORMATION region{};
        char module[MAX_PATH]{};
        if (VirtualQuery(address, &region, sizeof(region)))
            GetModuleFileNameA(static_cast<HMODULE>(region.AllocationBase), module, MAX_PATH);
        std::snprintf(line, sizeof(line), "  %p %s+0x%llx", address, module,
                      static_cast<unsigned long long>(pc - reinterpret_cast<DWORD64>(region.AllocationBase)));
        write_error(line);
    }
    write_error("\n");
}
#endif

void report(const char* kind, bool exception = true) noexcept {
    if (reporting.test_and_set()) {
        write_error("runtime_fatal error=recursive_fault\n");
        std::_Exit(127);
    }
    write_error("runtime_fatal error=");
    write_error(kind);
    write_error("\n");
    if (exception) {
        if (auto current = std::current_exception()) {
            try { std::rethrow_exception(current); }
            catch (const std::exception& error) {
                write_error("exception.what: ");
                write_error(error.what());
                write_error("\n");
            } catch (...) { write_error("exception: non-std exception\n"); }
        }
    }
    write_error("stacktrace:\n");
#ifdef _WIN32
    void* frames[64];
    const auto count = CaptureStackBackTrace(0, 64, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) print_address(frames[i]);
    FlushFileBuffers(GetStdHandle(STD_ERROR_HANDLE));
#else
    void* frames[64];
    const auto count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
#endif
}

[[noreturn]] void terminated() noexcept {
    report("terminate");
    std::_Exit(134);
}

#ifdef _WIN32
[[noreturn]] void invalid_parameter(const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {
    report("invalid_parameter");
    std::_Exit(134);
}

[[noreturn]] void purecall() {
    report("purecall");
    std::_Exit(134);
}

LONG WINAPI unhandled(EXCEPTION_POINTERS* exception) noexcept {
    char description[96];
    std::snprintf(description, sizeof(description), "unhandled_exception code=0x%08lx",
                  exception->ExceptionRecord->ExceptionCode);
    report(description);
    write_error("fault_address:\n");
    print_address(exception->ExceptionRecord->ExceptionAddress);
#if defined(_M_X64)
    auto context = *exception->ContextRecord;
    STACKFRAME64 frame{};
    frame.AddrPC = {context.Rip, 0, AddrModeFlat};
    frame.AddrStack = {context.Rsp, 0, AddrModeFlat};
    frame.AddrFrame = {context.Rbp, 0, AddrModeFlat};
    for (unsigned i = 0; i < 64 && symbols_ready; ++i) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, symbol_process, GetCurrentThread(), &frame,
                         &context, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) break;
        if (!frame.AddrPC.Offset) break;
        print_address(reinterpret_cast<void*>(frame.AddrPC.Offset));
    }
#endif
    FlushFileBuffers(GetStdHandle(STD_ERROR_HANDLE));
    return EXCEPTION_EXECUTE_HANDLER;
}
#else
void fatal_signal(int number) noexcept {
    const char* kind = number == SIGSEGV ? "SIGSEGV" : number == SIGABRT ? "SIGABRT" :
                       number == SIGILL ? "SIGILL" : number == SIGFPE ? "SIGFPE" : "SIGBUS";
    report(kind, false);
    std::_Exit(128 + number);
}
#endif
} // namespace

void log_runtime_error(const char* scope, const char* message) noexcept {
    char buffer[2048];
    std::snprintf(buffer, sizeof(buffer), "runtime_error scope=%s message=%s\n", scope, message);
    write_error(buffer);
}

void install_fatal_diagnostics() noexcept {
    std::set_terminate(terminated);
#ifdef _WIN32
    _set_invalid_parameter_handler(invalid_parameter);
    _set_purecall_handler(purecall);
    SetUnhandledExceptionFilter(unhandled);
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    // Restrict symbol lookup to the executable directory. Never block a crash
    // report on a symbol server, and load DbgHelp/PDB state before any failure.
    wchar_t directory[32768]{};
    GetModuleFileNameW(nullptr, directory, 32768);
    if (auto* slash = std::wcsrchr(directory, L'\\')) *slash = L'\0';
    DuplicateHandle(GetCurrentProcess(), GetCurrentProcess(), GetCurrentProcess(),
                    &symbol_process, 0, FALSE, DUPLICATE_SAME_ACCESS);
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_FAIL_CRITICAL_ERRORS | SYMOPT_NO_PROMPTS);
    symbols_ready = SymInitializeW(symbol_process, directory, TRUE) != FALSE;
#else
    void* warmup[1];
    backtrace(warmup, 1);
    static char signal_stack[64 * 1024];
    stack_t alternate{};
    alternate.ss_sp = signal_stack;
    alternate.ss_size = sizeof(signal_stack);
    sigaltstack(&alternate, nullptr);
    struct sigaction action{};
    action.sa_handler = fatal_signal;
    action.sa_flags = SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    for (int number : {SIGABRT, SIGSEGV, SIGILL, SIGFPE, SIGBUS}) sigaction(number, &action, nullptr);
#endif
#ifdef MFQ_ENABLE_FATAL_TESTS
    const char* test = std::getenv("MFQ_TEST_RUNTIME_FATAL");
    if (!test) return;
    if (std::strcmp(test, "terminate") == 0) {
        try { throw std::runtime_error("MFQ fatal diagnostic test exception"); }
        catch (...) { std::terminate(); }
    }
#ifdef _WIN32
    if (std::strcmp(test, "invalid_parameter") == 0) {
        char buffer[2];
        strcpy_s(buffer, sizeof(buffer), "invalid parameter test");
    }
    if (std::strcmp(test, "purecall") == 0) _purecall();
    if (std::strcmp(test, "access_violation") == 0)
        RaiseException(EXCEPTION_ACCESS_VIOLATION, 0, 0, nullptr);
#else
    if (std::strcmp(test, "signal") == 0) raise(SIGABRT);
#endif
#endif
}
} // namespace mfq
