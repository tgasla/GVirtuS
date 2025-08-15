/**
 * @file   HybridCommunicator.h
 * @author Xiaoyu Luo
 * @date   2025-08-01
 */
#pragma once

#include "gvirtus/communicators/Communicator.h"
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace gvirtus::communicators {

// Forward declarations to avoid including heavy headers here
class TcpCommunicator;
class RdmaCommunicator;

/** Available transport channels */
enum class Transport : uint8_t { TCP = 0, RDMA = 1 };

/**
 * Function-level transport selector.
 * Given a routine name (and optionally a bytes_hint),
 * returns the desired transport channel.
 * Returning std::nullopt means "undecided" — the default transport will be used.
 */
using TransportSelector =
    std::function<std::optional<Transport>(const std::string& routine,
                                           size_t bytes_hint)>;

/**
 * HybridCommunicator
 *
 * Combines two underlying channels (TCP and RDMA) into a single session,
 * exposing the standard Communicator interface to upper layers.
 *
 * The transport channel is chosen **per function call**:
 * all Read/Write operations within the same call go through the same channel.
 *
 * Usage pattern:
 *   - After writing the routine name on TCP, call begin_call(routine, Transport::RDMA/TCP, bytes_hint)
 *   - Perform all Read/Write for that call (Hybrid will steer header vs. payload)
 *   - Call end_call() when finished
 *
 * If begin_call() is not explicitly invoked:
 *   - If a selector is set, it will be used to choose the transport
 *   - Otherwise, the default transport will be used
 */
class HybridCommunicator : public Communicator {
 private:
  std::shared_ptr<Communicator> _tcp;
  std::shared_ptr<Communicator> _rdma;

  // Default transport when no explicit begin_call() and selector returns no result
  Transport _default_transport = Transport::TCP;

  // Optional auto selector (used when not explicitly forced)
  TransportSelector _selector;

  // Current call context (valid only within the lifetime of a single function call)
  std::mutex _ctx_mu;
  std::optional<Transport> _current_call_transport;
  std::string _current_call_routine;
  size_t _current_call_bytes_hint = 0;

  // --- Per-call header steering state (for RDMA calls only) ---
  // Ensure the first length header (8 bytes) of a call goes over TCP.
  bool   _awaiting_header_tx = false; // true until the 8B header has been written via TCP
  size_t _header_remaining_tx = 0;    // remaining header bytes to write on TCP

  bool   _awaiting_header_rx = false; // true until the 8B header has been read via TCP
  size_t _header_remaining_rx = 0;    // remaining header bytes to read from TCP

 public:
  HybridCommunicator() = default;

  // Construct with both channels already connected/ready
  HybridCommunicator(std::shared_ptr<Communicator> tcp,
                     std::shared_ptr<Communicator> rdma,
                     Transport default_transport = Transport::TCP)
      : _tcp(std::move(tcp)),
        _rdma(std::move(rdma)),
        _default_transport(default_transport) {}

  ~HybridCommunicator() override = default;

  // ---- Communicator interface ---- //
  void Serve() override;
  const Communicator *const Accept() const override;
  void Connect() override;
  size_t Read(char *buffer, size_t size) override;
  size_t Write(const char *buffer, size_t size) override;
  void Sync() override;
  void Close() override;
  std::string to_string() override { return "hybridcommunicator"; }

  // ---- Per-call transport control ---- //

  /**
   * Mark the start of a new function call and bind its transport channel.
   * bytes_hint helps downstream decisions, e.g., RDMA window sizing.
   */
  void begin_call(const std::string& routine, Transport t, size_t bytes_hint);

  /** Mark the end of the current function call and clear its context. */
  void end_call();

  /** Set the function-level selector (used when no explicit forced transport). */
  void set_selector(TransportSelector sel) { _selector = std::move(sel); }

  /** Set the default transport. */
  void set_default_transport(Transport t) { _default_transport = t; }

 private:
  // Decide which channel to use for the current call (may set _current_call_transport)
  std::shared_ptr<Communicator> current_channel_locked();
};

} // namespace gvirtus::communicators
