
#include <iostream>
#include <thread_pool.h>

namespace mp{

    //Initialize empty threadpool
    ThreadPool::ThreadPool():stop_flag_(false), task_counter_(0){}

    //Shut down all the threads 
    ThreadPool::~ThreadPool(){
        //Set stop flag and wake up all waiting threads
        stop_flag_.store(true);
        worker_condition_.notify_all();
    }

    //Initialize the threadpool with n threads
    bool ThreadPool::initialize(const int thread_count){
        //Reserve memory for the worker threads
        worker_threads_.reserve(thread_count);
        try{
            for(int i = 0; i < thread_count; ++i){
                worker_threads_.emplace_back(
                    //Capture the instance, launch the manager func in the thread
                    [this](std::stop_token token){management(token);}
                );
            }
            return true;
        }
        catch(const std::system_error& e){
            std::cerr << "Thread pool failed to create a thread: " << e.what();
        }
        return false;
    }

    //Retuns the number of work threads in the pool
    const int ThreadPool::pool_size(){
        return worker_threads_.size();
    }

    //Waits until task counter reaches zero
    void ThreadPool::wait_for_work(){
        //Queue lock to sync us with the pushing/popping
        std::unique_lock<std::mutex> queue_lock(work_queue_lock_);
        //Wait until task counter reaches zero
        worker_condition_.wait(queue_lock, [this](){return task_counter_.load() == 0;});
    }

    //Manager function for the worker threads
    void ThreadPool::management(std::stop_token token){
        //Wait for work until requested to stop
        while(!token.stop_requested()){
            std::function<void()> task;
            {   //Mutex scope, create lock for queue, shared between the threads
                std::unique_lock<std::mutex> queue_lock(work_queue_lock_);
                //Wait until notified of work or need to quit. queue_lock is released temporarily
                worker_condition_.wait(queue_lock, [this]{return stop_flag_.load() || !work_tasks_.empty();});
                //Reaquire the queue lock, see if we need to quit
                if(stop_flag_.load()){
                    return;
                }
                //Get task from queue
                task = std::move(work_tasks_.front());
                work_tasks_.pop();
            } // Release queue lock while we perform work
            task();
            //Decrement the task counter
            if(task_counter_.fetch_sub(1) == 1){
                std::lock_guard<std::mutex> lock(work_queue_lock_);
                //Wake up wait_for_work. Work threads need check
                //the queue and go back to sleep
                worker_condition_.notify_all(); 
            }
        }
    }

}