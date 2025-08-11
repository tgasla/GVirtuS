/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   TcpCommunicator.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 8 12:08:33 2009
 *
 * @brief
 *
 *
 */
// #define DEBUG

#include "TcpCommunicator.h"

#ifndef _WIN32

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#else
#include <WinSock2.h>
static bool initialized = false;
#endif

#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Rdma.h>
#include <gvirtus/communicators/Endpoint_Tcp.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;
using gvirtus::communicators::TcpCommunicator;

TcpCommunicator::TcpCommunicator(const std::string &communicator) {
#ifdef _WIN32
    if (!initialized) {
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
            throw runtime_error("Cannot initialized WinSock.");
        initialized = true;
    }
#endif

    const char *valueptr = strstr(communicator.c_str(), "://") + 3;
    const char *portptr = strchr(valueptr, ':');
    if (portptr == NULL) throw runtime_error("Port not specified.");
    mPort = (short)strtol(portptr + 1, NULL, 10);

#ifdef _WIN32
    char *hostname = _strdup(valueptr);
#else
    char *hostname = strdup(valueptr);
#endif

    hostname[portptr - valueptr] = 0;
    mHostname = string(hostname);
    struct hostent *ent = gethostbyname(hostname);
    free(hostname);
    if (ent == NULL)
        throw runtime_error("TcpCommunicator: Can't resolve hostname '" + mHostname + "'.");
    mInAddrSize = ent->h_length;
    mInAddr = new char[mInAddrSize];
    memcpy(mInAddr, *ent->h_addr_list, mInAddrSize);
}

TcpCommunicator::TcpCommunicator(const char *hostname, short port) {
    mHostname = string(hostname);
    struct hostent *ent = gethostbyname(hostname);
    if (ent == NULL)
        throw runtime_error("TcpCommunicator: Can't resolve hostname '" + mHostname + "'.");
    mInAddrSize = ent->h_length;
    mInAddr = new char[mInAddrSize];
    memcpy(mInAddr, *ent->h_addr_list, mInAddrSize);
    mPort = port;
}

TcpCommunicator::TcpCommunicator(int fd, const char *hostname) {
    mSocketFd = fd;
    InitializeStream();
}

TcpCommunicator::~TcpCommunicator() {
    //    close(mSocketFd);
    delete[] mInAddr;
}

void TcpCommunicator::Serve() {
#ifdef DEBUG
    printf("TcpCommunicator::Serve() called\n");
#endif

    struct sockaddr_in socket_addr;

    if ((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        throw runtime_error("TcpCommunicator: Can't create socket: " + string(strerror(errno)) +
                            ".");

    memset((char *)&socket_addr, 0, sizeof(struct sockaddr_in));

    socket_addr.sin_family = AF_INET;
    socket_addr.sin_port = htons(mPort);
    socket_addr.sin_addr.s_addr = INADDR_ANY;

    char on = 1;
    setsockopt(mSocketFd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    int bindResult = bind(mSocketFd, (struct sockaddr *)&socket_addr, sizeof(struct sockaddr_in));
    if (bindResult != 0)
        throw runtime_error("TcpCommunicator: Can't bind socket: " + string(strerror(errno)) + ".");

    int listenResult = listen(mSocketFd, 5);
    if (listenResult != 0)
        throw runtime_error(
            "TcpCommunicator: Can't listen from socket: " + string(strerror(errno)) + ".");

#ifdef DEBUG
    cout << "TcpCommunicator::Serve() returned" << endl;
#endif
}

const gvirtus::communicators::Communicator *const TcpCommunicator::Accept() const {
#ifdef DEBUG
    cout << "TcpCommunicator::Accept() called" << endl;
#endif

    unsigned client_socket_fd;
    struct sockaddr_in client_socket_addr;
#ifndef _WIN32
    unsigned client_socket_addr_size;
#else
    int client_socket_addr_size;
#endif

    client_socket_addr_size = sizeof(struct sockaddr_in);
    if ((client_socket_fd =
             accept(mSocketFd, (sockaddr *)&client_socket_addr, &client_socket_addr_size)) == 0 ||
        errno == EINTR) {
        return nullptr;
    }

#ifdef DEBUG
    cout << "TcpCommunicator::Accept() client_socket_fd: " << client_socket_fd << endl;
#endif
    return new TcpCommunicator(client_socket_fd, inet_ntoa(client_socket_addr.sin_addr));
}

void TcpCommunicator::Connect() {
#ifdef DEBUG
    cout << "TcpCommunicator::Connect() called " < < < < endl;
#endif

    struct sockaddr_in remote;

    if ((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        throw runtime_error("TcpCommunicator: Can't create socket: " + string(strerror(errno)) +
                            ".");

    remote.sin_family = AF_INET;
    remote.sin_port = htons(mPort);
    memcpy(&remote.sin_addr, mInAddr, mInAddrSize);

    if (connect(mSocketFd, (struct sockaddr *)&remote, sizeof(struct sockaddr_in)) != 0)
        throw runtime_error("TcpCommunicator: Can't connect to socket: " + string(strerror(errno)) +
                            ".");

    InitializeStream();

#ifdef DEBUG
    cout << "TcpCommunicator::Connect() returned" << endl;
#endif
}

void TcpCommunicator::Close() {}

size_t TcpCommunicator::Read(char *buffer, size_t size) {
#ifdef DEBUG
    cout << "TcpCommunicator::Read() size: " << size << endl;
#endif

    mpInput->read(buffer, size);

#ifdef DEBUG
    for (unsigned int i = 0; i < size; i++) printf("%d LETTO %02X\n", i, buffer[i]);
#endif

    size_t ret_value;
    if (mpInput->bad() || mpInput->eof())
        ret_value = 0;
    else
        ret_value = size;

#ifdef DEBUG
    cout << "TcpCommunicator::Read() returned " << ret_value << endl;
#endif

    return ret_value;
}

size_t TcpCommunicator::Write(const char *buffer, size_t size) {
#ifdef DEBUG
    cout << "TcpCommunicator::Write() called" << endl;
#endif

    mpOutput->write(buffer, size);

#ifdef DEBUG
    for (unsigned int i = 0; i < size; i++) printf("%d SCRITTO %02X \n", i, buffer[i]);
#endif

#ifdef DEBUG
    cout << "TcpCommunicator::Write() returned" << size << endl;
#endif

    return size;
}

void TcpCommunicator::Sync() { mpOutput->flush(); }

void TcpCommunicator::InitializeStream() {
#ifdef _WIN32
    FILE *i = _fdopen(mSocketFd, "r");
    FILE *o = _fdopen(mSocketFd, "w");
    mpInputBuf = new filebuf(i);
    mpOutputBuf = new filebuf(o);
#else
    mpInputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd, ios_base::in);
    mpOutputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd, ios_base::out);
#endif

    mpInput = new istream(mpInputBuf);
    mpOutput = new ostream(mpOutputBuf);
}

extern "C" std::shared_ptr<TcpCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    std::string arg =
        "tcp://" + std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Tcp>(end)->address() +
        ":" +
        std::to_string(
            std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Tcp>(end)->port());
    return std::make_shared<TcpCommunicator>(arg);
}
