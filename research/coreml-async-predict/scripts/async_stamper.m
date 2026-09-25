// PB-ASYNC-STAMPED, Phase 0 only (research): the asynchronous Core ML predict with the native
// completion stamped before the Python completion handler runs.
//
// +[LayaAsyncStamper predict:features:options:handler:] sends
// predictionFromFeatures:options:completionHandler: with a block that stamps the clock on entry
// (the moment Core ML calls the completion, on its own thread) and then calls the Python handler
// block PyObjC built. scripts/prebind_async.py compiles this file with the system clang into a
// temporary dylib, loads it with ctypes.CDLL (which registers the class), and reads the last
// stamp with laya_async_native_stamp() after the waiter resumes. One predict is in flight at a
// time in Phase 0, so one slot is enough. It builds and copies nothing else.
//
// The clock is #83's predict_options_stamped.m's: mach_absolute_time() scaled by the timebase,
// the same arithmetic CPython's time.monotonic_ns() uses on macOS.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <stdint.h>

static mach_timebase_info_data_t timebase;
static volatile uint64_t last_native_ns;

static inline uint64_t now_ns(void) {
    uint64_t t = mach_absolute_time();
    return (t / timebase.denom) * timebase.numer + (t % timebase.denom) * timebase.numer / timebase.denom;
}

__attribute__((constructor)) static void init_timebase(void) { mach_timebase_info(&timebase); }

uint64_t laya_async_native_stamp(void) { return __atomic_load_n(&last_native_ns, __ATOMIC_SEQ_CST); }

@interface LayaAsyncStamper : NSObject
+ (void)predict:(MLModel *)model
       features:(id<MLFeatureProvider>)features
        options:(MLPredictionOptions *)options
        handler:(void (^)(id<MLFeatureProvider>, NSError *))handler;
@end

@implementation LayaAsyncStamper
+ (void)predict:(MLModel *)model
       features:(id<MLFeatureProvider>)features
        options:(MLPredictionOptions *)options
        handler:(void (^)(id<MLFeatureProvider>, NSError *))handler {
    [model predictionFromFeatures:features
                          options:options
                completionHandler:^(id<MLFeatureProvider> output, NSError *error) {
                  __atomic_store_n(&last_native_ns, now_ns(), __ATOMIC_SEQ_CST);
                  handler(output, error);
                }];
}
@end
