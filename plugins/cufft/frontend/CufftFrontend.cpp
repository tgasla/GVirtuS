/*
 * Written By: Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>,
 *             Department of Science and Technologies
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            School of Computer Science, University College Dublin
 */

#include "CufftFrontend.h"

using namespace std;

using gvirtus::common::mappedPointer;
using gvirtus::frontend::Frontend;

CufftFrontend msInstance __attribute_used__;

map<const void*, mappedPointer>* CufftFrontend::mappedPointers = NULL;
set<const void*>* CufftFrontend::devicePointers = NULL;
map<pthread_t, stack<void*>*>* CufftFrontend::toManage = NULL;

CufftFrontend::CufftFrontend() {
    if (devicePointers == NULL) devicePointers = new set<const void*>();
    if (mappedPointers == NULL) mappedPointers = new map<const void*, mappedPointer>();
    if (toManage == NULL) toManage = new map<pthread_t, stack<void*>*>();
    Frontend::GetFrontend();
}

CufftFrontend::~CufftFrontend() {}
