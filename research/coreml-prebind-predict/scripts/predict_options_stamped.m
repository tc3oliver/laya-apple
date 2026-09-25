// Core ML predict with prediction options and native timestamps on both sides (research only).
//
// This is research/coreml-nogil-product-mix/scripts/predict_stamped.m (#77) with one extra
// argument: the MLPredictionOptions that carry PB's output backings. scripts/prebind.py
// compiles it with the system clang into a temporary dylib and calls it through ctypes.CDLL,
// which releases the GIL once for the call. It sends predictionFromFeatures:options:error:
// and stamps the clock right before and right after it. It builds nothing and copies nothing:
// the model, the feature provider, the options and every MLMultiArray are created by PyObjC
// at load time.
//
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

void *laya_predict_options_stamped(void *model, void *provider, void *options, void **error, uint64_t *stamps) {
    NSError *err = nil;
    stamps[0] = now_ns();
    id<MLFeatureProvider> out = [(MLModel *)model predictionFromFeatures:(id<MLFeatureProvider>)provider
                                                                 options:(MLPredictionOptions *)options
                                                                   error:&err];
    stamps[1] = now_ns();
    if (error) *error = (void *)err;
    return (void *)out;
}
