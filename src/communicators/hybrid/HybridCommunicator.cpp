#include "HybridCommunicator.h"

#include <stdexcept>
#include <utility>
#include <iostream>
#include <algorithm> // std::min

#include "gvirtus/communicators/Communicator.h"
#include "gvirtus/communicators/Endpoint.h"
#include "gvirtus/communicators/Endpoint_Hybrid.h"

// Optional: direct access to concrete types if needed elsewhere
#include "../tcp/TcpCommunicator.h"
#include "../rdma/RdmaCommunicator.h"

using gvirtus::communicators::HybridCommunicator;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::Endpoint;
using gvirtus::communicators::Endpoint_Hybrid;
using gvirtus::communicators::Transport;

// --- First length header size (aligned with Buffer's framing) ---
static constexpr size_t kHeaderSize = sizeof(size_t);

// ----------------------
// Private helper
// ----------------------
std::shared_ptr<Communicator> HybridCommunicator::current_channel_locked() {
  // If the transport for this call is not decided yet:
  if (!_current_call_transport.has_value()) {
    // 1) Try selector if provided and routine is known
    if (_selector && !_current_call_routine.empty()) {
      auto decided = _selector(_current_call_routine, _current_call_bytes_hint);
      if (decided.has_value()) {
        _current_call_transport = decided.value();
      }
    }
    // 2) Fall back to default
    if (!_current_call_transport.has_value()) {
      _current_call_transport = _default_transport;
    }
  }

  // Return the bound channel
  switch (_current_call_transport.value()) {
    case Transport::TCP:
      if (!_tcp) throw std::runtime_error("HybridCommunicator: TCP channel is not available.");
      return _tcp;
    case Transport::RDMA:
      if (!_rdma) throw std::runtime_error("HybridCommunicator: RDMA channel is not available.");
      return _rdma;
    default:
      throw std::runtime_error("HybridCommunicator: Unknown transport selection.");
  }
}

// ----------------------
// begin_call / end_call （per-call context）
// ----------------------
void HybridCommunicator::begin_call(const std::string& routine,
                                    Transport t,
                                    size_t bytes_hint) {
  std::lock_guard<std::mutex> lk(_ctx_mu);
  _current_call_routine     = routine;
  _current_call_transport   = t;
  _current_call_bytes_hint  = bytes_hint;

  // Initialize "first 8B header goes over TCP" (only meaningful for RDMA + payload)
  if (t == Transport::RDMA && bytes_hint > 0) {
    _awaiting_header_tx = true;
    _header_remaining_tx = kHeaderSize;

    _awaiting_header_rx = true;
    _header_remaining_rx = kHeaderSize;
  } else {
    _awaiting_header_tx = false;
    _header_remaining_tx = 0;
    _awaiting_header_rx = false;
    _header_remaining_rx = 0;
  }

#ifdef DEBUG
  std::cout << "[Hybrid] begin_call: routine=" << routine
            << " transport=" << (t==Transport::RDMA?"RDMA":"TCP")
            << " bytes_hint=" << bytes_hint
            << " header8->TCP=" << ((t==Transport::RDMA && bytes_hint>0)?"on":"off")
            << std::endl;
#endif
}

void HybridCommunicator::end_call() {
  std::lock_guard<std::mutex> lk(_ctx_mu);

#ifdef DEBUG
  std::cout << "[Hybrid] end_call: routine=" << _current_call_routine
            << std::endl;
#endif

  _current_call_routine.clear();
  _current_call_transport.reset();
  _current_call_bytes_hint = 0;

  // Clear header steering state
  _awaiting_header_tx = false;
  _header_remaining_tx = 0;
  _awaiting_header_rx = false;
  _header_remaining_rx = 0;
}

// ----------------------
// Communicator interface
// ----------------------
void HybridCommunicator::Serve() {
#ifdef DEBUG
  std::cout << "[Hybrid] Serve()" << std::endl;
#endif
  if (_tcp) _tcp->Serve();
  if (_rdma) _rdma->Serve();
}

const Communicator *const HybridCommunicator::Accept() const {
#ifdef DEBUG
  std::cout << "[Hybrid] Accept()" << std::endl;
#endif

  // Accept from both listeners; we assume pairing is handled at the endpoint layer.
  const Communicator* tcp_raw  = _tcp  ? _tcp->Accept()  : nullptr;
  const Communicator* rdma_raw = _rdma ? _rdma->Accept() : nullptr;

  if (!tcp_raw || !rdma_raw) {
#ifdef DEBUG
    std::cerr << "[Hybrid] Accept(): missing one of the channels (tcp="
              << (tcp_raw ? "ok" : "null") << ", rdma="
              << (rdma_raw ? "ok" : "null") << ")" << std::endl;
#endif
    return nullptr;
  }

  // Wrap with shared_ptr but do not take ownership of the raw pointers returned by Accept()
  auto noop = [](Communicator*){};
  std::shared_ptr<Communicator> tcp_sp(const_cast<Communicator*>(tcp_raw), noop);
  std::shared_ptr<Communicator> rdma_sp(const_cast<Communicator*>(rdma_raw), noop);

  // Create a per-session Hybrid with the two accepted channels
  auto* child = new HybridCommunicator(tcp_sp, rdma_sp, _default_transport);
  return child;
}

void HybridCommunicator::Connect() {
#ifdef DEBUG
  std::cout << "[Hybrid] Connect()" << std::endl;
#endif
  if (_tcp) _tcp->Connect();
  if (_rdma) _rdma->Connect();
}

size_t HybridCommunicator::Read(char *buffer, size_t size) {
  if (!buffer || size == 0) return 0;

  size_t total_read = 0;

  // Step 1: If this call uses RDMA and we still owe the first 8B header,
  //         read that header from TCP first.
  {
    std::lock_guard<std::mutex> lk(_ctx_mu);

    if (_current_call_transport.has_value() &&
        _current_call_transport.value() == Transport::RDMA &&
        _awaiting_header_rx && _header_remaining_rx > 0) {

      const size_t take = std::min(size, _header_remaining_rx);
      const size_t got  = _tcp->Read(buffer, take);
      total_read += got;
      _header_remaining_rx -= got;

#ifdef DEBUG
      std::cout << "[Hybrid][Read] header(TCP) got=" << got
                << " remain=" << _header_remaining_rx << std::endl;
#endif
      if (_header_remaining_rx == 0) {
        _awaiting_header_rx = false; // subsequent bytes belong to payload
      }

      if (total_read == size) return total_read;

      // Else, fall through to read remaining bytes (payload) on bound channel
    }
  }

  // Step 2: Remaining bytes -> bound channel (RDMA/TCP)
  std::shared_ptr<Communicator> ch;
  {
    std::lock_guard<std::mutex> lk(_ctx_mu);
    ch = current_channel_locked();
  }

  total_read += ch->Read(buffer + total_read, size - total_read);
#ifdef DEBUG
  std::cout << "[Hybrid][Read] body("
            << (ch.get()==_rdma.get()?"RDMA":"TCP")
            << ") got=" << (total_read) << std::endl;
#endif
  return total_read;
}

size_t HybridCommunicator::Write(const char *buffer, size_t size) {
  if (!buffer || size == 0) return 0;

  size_t total_written = 0;

  // Step 1: If this call uses RDMA and we still owe the first 8B header,
  //         write that header via TCP first.
  {
    std::lock_guard<std::mutex> lk(_ctx_mu);

    if (_current_call_transport.has_value() &&
        _current_call_transport.value() == Transport::RDMA &&
        _awaiting_header_tx && _header_remaining_tx > 0) {

      const size_t take = std::min(size, _header_remaining_tx);
      const size_t put  = _tcp->Write(buffer, take);
      total_written += put;
      _header_remaining_tx -= put;

#ifdef DEBUG
      std::cout << "[Hybrid][Write] header(TCP) put=" << put
                << " remain=" << _header_remaining_tx << std::endl;
#endif
      
      if (_header_remaining_tx == 0) {
  _awaiting_header_tx = false; // 后续才是 payload

  // 🔧 关键：头已经写完，立刻把 TCP 刷掉，保证对端能读到 8B
  if (_tcp) _tcp->Sync();
#ifdef DEBUG
  std::cout << "[Hybrid][Write] header(TCP) flushed" << std::endl;
#endif
}
      if (total_written == size) return total_written;

      buffer += put;
      size   -= put;
    }
  }

  // Step 2: Remaining bytes -> bound channel (RDMA/TCP)
  std::shared_ptr<Communicator> ch;
  {
    std::lock_guard<std::mutex> lk(_ctx_mu);
    ch = current_channel_locked();
  }

  total_written += ch->Write(buffer, size);
#ifdef DEBUG
  std::cout << "[Hybrid][Write] body("
            << (ch.get()==_rdma.get()?"RDMA":"TCP")
            << ") put=" << (total_written) << std::endl;
#endif
  return total_written;
}

void HybridCommunicator::Sync() {
#ifdef DEBUG
  std::cout << "[Hybrid] Sync()" << std::endl;
#endif
  std::lock_guard<std::mutex> lk(_ctx_mu);

  if (_current_call_transport.has_value() &&
      _current_call_transport.value() == Transport::RDMA) {
    // call RDMA：head in TCP，payload in RDMA —— flush both channels
    if (_tcp)  _tcp->Sync();
    if (_rdma) _rdma->Sync();
  } else {
    // not choose portocal or it is TCP this round：only flush TCP
    if (_tcp) _tcp->Sync();
  }
}


void HybridCommunicator::Close() {
#ifdef DEBUG
  std::cout << "[Hybrid] Close()" << std::endl;
#endif
  if (_tcp) _tcp->Close();
  if (_rdma) _rdma->Close();

  // Clear call context
  {
    std::lock_guard<std::mutex> lk(_ctx_mu);
    _current_call_routine.clear();
    _current_call_transport.reset();
    _current_call_bytes_hint = 0;

    // Clear header steering state
    _awaiting_header_tx = false;
    _header_remaining_tx = 0;
    _awaiting_header_rx = false;
    _header_remaining_rx = 0;
  }
}

// ----------------------
// Optional factory
// ----------------------
extern "C" std::shared_ptr<gvirtus::communicators::HybridCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {

  auto hybrid_ep = std::dynamic_pointer_cast<Endpoint_Hybrid>(end);
  if (!hybrid_ep) {
    throw std::runtime_error("HybridCommunicator factory: invalid endpoint type (expected Endpoint_Hybrid).");
  }

  // Build underlying communicators from the endpoint
  auto tcp = std::make_shared<gvirtus::communicators::TcpCommunicator>(
      hybrid_ep->address().c_str(), static_cast<short>(hybrid_ep->tcp_port()));

  auto rdma = std::make_shared<gvirtus::communicators::RdmaCommunicator>(
      hybrid_ep->address(), std::to_string(hybrid_ep->rdma_port()),
      /*isRoce*/ hybrid_ep->rdma_suite() == "roce-rdma");

  auto hc = std::make_shared<HybridCommunicator>(tcp, rdma, Transport::TCP);
  return hc;
}
