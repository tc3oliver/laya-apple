// Core ML predict with native timestamps on both sides of the call (research only).
//
// scripts/nogil.py compiles this with the system clang into a temporary dylib and calls it
// through ctypes.CDLL, which releases the GIL for the call. It sends the same Objective-C
// message PyObjC sends in research/coreml-gil-completion-path/scripts/nogil_predict.py
// (predictionFromFeatures:error:), and stamps the clock right before and right after it.
// The stamps are time.monotonic_ns() values: mach_absolute_time() scaled by the timebase,
// the same arithmetic CPython uses on macOS. The caller stamps again once ctypes has
// re-acquired the GIL, so stamps[1] -> that stamp is the ANE thread's GIL re-acquire wait.
//
// Manual reference counting (-fno-objc-arc): the prediction and the error come back
// autoreleased, into the autorelease pool the Python caller holds open.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <stdint.h>

static mach_timebase_info_data_t timebase;

static inline uint64_t now_ns(void) {
    uint64_t t = mach_absolute_time();
    return (t / timebase.denom) * timebase.numer + (t % timebase.denom) * timebase.numer / timebase.denom;
}

__attribute__((constructor)) static void init_timebase(void) { mach_timebase_info(&timebase); }

void *laya_predict_stamped(void *model, void *provider, void **error, uint64_t *stamps) {
    NSError *err = nil;
    stamps[0] = now_ns();
    id<MLFeatureProvider> out = [(MLModel *)model predictionFromFeatures:(id<MLFeatureProvider>)provider error:&err];
    stamps[1] = now_ns();
    if (error) *error = (void *)err;
    return (void *)out;
}
