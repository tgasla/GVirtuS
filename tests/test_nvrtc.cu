/*
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             School of Computer Science, University College Dublin
 */

#include <gtest/gtest.h>
#include <nvrtc.h>

#define NVRTC_CHECK(err) ASSERT_EQ((err), NVRTC_SUCCESS)

TEST(nvrtc, versionCheck) {
    int major = 0, minor = 0;
    nvrtcVersion(&major, &minor);
    ASSERT_EQ(major, 0);  // I have not implemented this, so it should be 0
    ASSERT_EQ(minor, 0);  // I have not implemented this, so it should be 0
}
