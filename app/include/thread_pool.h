#ifndef MP_COURSE_THREAD_POOL_H
#define MP_COURSE_THREAD_POOL_H

#include <queue>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include <condition_variable>

namespace mp_course{
    class ThreadPool{
        public:
            //Initialize empty threadpool
            ThreadPool();

            //Shut down all the threads 
            ~ThreadPool();

            //Initialize the threadpool with n threads
            bool initialize(const int thread_count);

            //Queue's work for the threadpool to execute
            template <typename Function, typename... Arguments>
            void queue_work(Function&& work, Arguments&&... arguments){
                {//Acquire the queue lock
                    std::unique_lock<std::mutex> queue_lock(work_queue_lock_);
                    //push new work task and wake up one thread to deal with it
                    work_tasks_.push(
                        //Capture the function as forward. Expand the arguments into tuple (maintain right values, while allowing left values)
                        [
                            func = std::forward<Function>(work),
                            args = std::make_tuple(std::forward<Arguments>(arguments)...)
                        ]() mutable {
                            //Pass the arguments to the function
                            std::apply(func, args);
                        }
                    );
                    task_counter_.fetch_add(1);
                } //Release the queue lock, before waking up a thread
                worker_condition_.notify_one();
            }

            //Retuns the number of work threads in the pool
            const int pool_size();

            //Waits until task counter reaches zero
            void wait_for_work();

        private:
            std::vector<std::jthread> worker_threads_;
            std::queue<std::function<void()>> work_tasks_;
            std::condition_variable worker_condition_;
            std::mutex work_queue_lock_;
            std::atomic<bool> stop_flag_;
            std::atomic<int> task_counter_;

            //Manager function for the worker threads
            void management(std::stop_token token);
    };
}

#endif