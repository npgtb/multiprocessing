#ifndef MP_COURSE_THREAD_POOL_H
#define MP_COURSE_THREAD_POOL_H

#include <cmath>
#include <queue>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include <condition_variable>

namespace mp{
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

    //Generic function for queueing linear work into the threadpool
    template <typename Function, typename... Arguments>
    void queue_linear_work(
        ThreadPool& thread_pool, int work_size,
        Function&& work, Arguments&&... arguments
    ){
        if(work_size > 0){
            //Calculate work size
            const int thread_count = std::min(work_size, thread_pool.pool_size());
            const int work_chunk_size = std::floor(work_size /  thread_count);
            //Queue work
            for(int i = 0; i < (thread_count-1); ++i){
                const int chunk_start = (work_chunk_size * i);
                const int chunk_end = (work_chunk_size * (i+1));
                thread_pool.queue_work(
                    std::forward<Function>(work),
                    std::forward<Arguments>(arguments)...,
                    chunk_start, chunk_end
                );
            }
            //Queue the remainder of the work into last thread
            const int remainder_start = work_chunk_size * (thread_count - 1);
            thread_pool.queue_work(
            std::forward<Function>(work),
                std::forward<Arguments>(arguments)...,
                remainder_start, work_size
            );
        }
    }
}

#endif