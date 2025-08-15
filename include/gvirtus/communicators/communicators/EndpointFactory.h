#pragma once

#include <gvirtus/common/JSON.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "Endpoint.h"
#include "Endpoint_Rdma.h"
#include "Endpoint_Tcp.h"
#include "Endpoint_Hybrid.h"

//#define DEBUG

namespace gvirtus::communicators {
class EndpointFactory {
 public:
  static std::shared_ptr<Endpoint> get_endpoint(const fs::path &json_path) {
#ifdef DEBUG
      std::cout << "EndpointFactory::get_endpoint() called" << std::endl;
#endif

    std::shared_ptr<Endpoint> ptr;
    std::ifstream ifs(json_path);
    nlohmann::json j;
    ifs >> j;

    // FIXME: This if-else smells...
    // tcp/ip
    if ("tcp/ip" == j["communicator"][ind_endpoint]["endpoint"].at("suite")) {
#ifdef DEBUG
        std::cout << "EndpointFactory::get_endpoint() found tcp/ip endpoint" << std::endl;
#endif
        auto end = common::JSON<Endpoint_Tcp>(json_path).parser();
        ptr = std::make_shared<Endpoint_Tcp>(end);
    }
    // infiniband
    else if ("infiniband-rdma" == j["communicator"][ind_endpoint]["endpoint"].at("suite")) {
#ifdef DEBUG
        std::cout << "EndpointFactory::get_endpoint() found infiniband endpoint" << std::endl;
#endif
        auto end = common::JSON<Endpoint_Rdma>(json_path).parser();
        ptr = std::make_shared<Endpoint_Rdma>(end);
    }
    else if ("roce-rdma" == j["communicator"][ind_endpoint]["endpoint"].at("suite")) {
#ifdef DEBUG
    std::cout << "EndpointFactory::get_endpoint() found rdma-roce endpoint (reusing Endpoint_Rdma)" << std::endl;
#endif
    auto end = common::JSON<Endpoint_Rdma>(json_path).parser();
    ptr = std::make_shared<Endpoint_Rdma>(end);
}
    else if ("hybrid" == j["communicator"][ind_endpoint]["endpoint"].at("suite")) {
#ifdef DEBUG
    std::cout << "EndpointFactory::get_endpoint() found hybrid endpoint" << std::endl;
#endif
    auto end = common::JSON<Endpoint_Hybrid>(json_path).parser();
    ptr = std::make_shared<Endpoint_Hybrid>(end);
}
    else {
        throw std::runtime_error("EndpointFactory::get_endpoint(): Your suite is not compatible!");
    }

    ind_endpoint++;

    j.clear();
    ifs.close();

#ifdef DEBUG
    std::cout << "EndpointFactoru::get_endpoint(): end is: " << ptr->to_string() << std::endl;
    std::cout << "EndpointFactory::get_endpoint() ended" << std::endl;
#endif

    return ptr;
  }

  static int index() { return ind_endpoint; }

 private:
  static int ind_endpoint;
};
}  // namespace gvirtus::communicators
