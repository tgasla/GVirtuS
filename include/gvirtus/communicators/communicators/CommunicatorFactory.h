#pragma once

#include <gvirtus/common/LD_Lib.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <stdlib.h> /* getenv */

#include "Communicator.h"
#include "Endpoint.h"
#include "Endpoint_Tcp.h"
#include "Endpoint_Rdma.h"
//#define DEBUG

namespace gvirtus::communicators {
    class CommunicatorFactory {
    public:
        static
        std::shared_ptr<common::LD_Lib<Communicator, std::shared_ptr<Endpoint>>>
        get_communicator(
                std::shared_ptr<Endpoint> end,
                bool secure = false
        ) {
#ifdef DEBUG
            std::cout << "CommunicatorFactory::get_communicator() called." << std::endl;
            std::cout << "CommunicatorFactory::get_communicator(): endpoint is: " << end->to_string() << std::endl;
#endif
            std::shared_ptr<common::LD_Lib<Communicator, std::shared_ptr<Endpoint>>> dl;

            std::string gvirtus_home = CommunicatorFactory::getGVirtuSHome();

#ifdef DEBUG
            std::cout << "CommunicatorFactory::get_communicator(): found gvirtus home: " << gvirtus_home << std::endl;
#endif
std::cout << "DEBUG: protocol string is [" << end->protocol() << "]" << std::endl;
            // Supported unsecure communicators
            std::vector<std::string> unsecureMatches = {"tcp",
                                                        "http",
                                                        "oldtcp",
                                                        "ws",
                                                        "ib",
                                                        "hybrid",
                                                        
                                                       
            };

            // Supported secure communicators
            std::vector<std::string> secureMatches = {"https",
                                                      "wss"
            };

            // Is the desired communicator a secure communicator?
            if (not secure) {
                // No, then search it into the supported unsecure communicators vector
                auto foundIt = std::find(unsecureMatches.begin(), unsecureMatches.end(), end->protocol());
                // Is the desired communicator supported?
                if (foundIt == unsecureMatches.end()) {
                    // No, then throw an exception.
                    throw std::runtime_error("Unsecure communicator not supported");
                }
            }
            else {
                // Yes, then search it into the supported secure communicators vector
                auto foundIt = std::find(secureMatches.begin(), secureMatches.end(), end->protocol());
                // Is the desired communicator supported?
                if (foundIt == secureMatches.end()) {
                    // No, then throw an exception.
                    throw std::runtime_error("Secure communicator not supported");
                }
            }

#ifdef DEBUG
            std::cout << "CommunicatorFactory::get_communicator(): found protocol: " << end->protocol() << std::endl;
#endif

            std::string dl_string = gvirtus_home + "/lib/libgvirtus-communicators-" + end->protocol() + ".so";

#ifdef DEBUG
            std::cout << "CommunicatorFactory::get_communicator(): dl_string: " << dl_string << std::endl;
#endif
            // dl is a ptr to an LD_Lib<Communicator, *Endpoint>
            dl =
                    std::make_shared<
                            common::LD_Lib<
                                    Communicator,
                                    std::shared_ptr<Endpoint>
                            >
                    >
                    (dl_string , "create_communicator");

#ifdef DEBUG
            std::cout << "CommunicatorFactory::get_communicator(): made dl" << std::endl;
#endif

            dl->build_obj(end);

#ifdef DEBUG
            std::cout << "CommunicatorFactory::get_communicator() ended" << std::endl;
#endif
            return dl;
        }

    private:
        static std::string getEnvVar(std::string const &key) {
            char *val = getenv(key.c_str());
            return val == NULL ? std::string("") : std::string(val);
        }

        static std::string getGVirtuSHome() {
            std::string gvirtus_home = CommunicatorFactory::getEnvVar("GVIRTUS_HOME");
            return gvirtus_home;
        }

        static std::string getConfigFile() {
            // Get the GVIRTUS_CONFIG environment varibale
            std::string config_path = CommunicatorFactory::getEnvVar("GVIRTUS_CONFIG");

            // Check if the configuration file is defined
            if (config_path == "") {
                // Check if the configuration file is in the GVIRTUS_HOME directory
                config_path = CommunicatorFactory::getEnvVar("GVIRTUS_HOME") + "/etc/properties.json";

                if (config_path == "") {
                    // Finally consider the current directory
                    config_path = "./properties.json";
                }
            }
            return config_path;
        }
    };
}  // namespace gvirtus::communicators
