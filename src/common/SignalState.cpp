#include <gvirtus/common/SignalException.h>
#include <gvirtus/common/SignalState.h>

#include <csignal>

using gvirtus::common::SignalState;

void SignalState::setup_signal_state(int signo) {
  struct sigaction a;
  a.sa_handler = true_flag;
  a.sa_flags = 0;
  sigemptyset(&a.sa_mask);

  if (sigaction(signo, &a, nullptr) < 0)
    throw SignalException("Impossibile installare l'handler per il segnale!");

  _signals_state.insert(std::make_pair(signo, false));
}

std::map<int, bool> SignalState::_signals_state = {};
