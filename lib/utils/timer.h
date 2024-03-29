#ifndef _TIMER_H_
#define _TIMER_H_

#include <time.h>

#ifdef _WIN32
#include <windows.h>
#if !defined(_WINSOCK2API_) && !defined(_WINSOCKAPI_)
struct timeval {
	long tv_sec;
	long tv_usec;
};
#endif
#else//_WIN32
#include <sys/time.h>
#endif//_WIN32


typedef double timer_dt;

class Timer
{
public:
	Timer();
	~Timer() {};

	void start();
	void stop();
	timer_dt get_time();
	timer_dt stop_get();
	timer_dt stop_get_start();

#ifdef _WIN32
	double freq;
	LARGE_INTEGER start_time;
	LARGE_INTEGER finish_time;
#else//_WIN32
	struct timeval start_time;
	struct timeval finish_time;
#endif//_WIN32
};


// Definition
#ifdef _WIN32
int gettimeofday(struct timeval* tv, int t) {
	union {
		long long ns100;
		FILETIME ft;
	} now;

	GetSystemTimeAsFileTime(&now.ft);
	tv->tv_usec = (long)((now.ns100 / 10LL) % 1000000LL);
	tv->tv_sec = (long)((now.ns100 - 116444736000000000LL) / 10000000LL);
	return (0);
}// gettimeofday()
#endif//_WIN32


Timer::Timer()
{
#ifdef _WIN32
	LARGE_INTEGER tmp;
	QueryPerformanceFrequency((LARGE_INTEGER*)& tmp);
	freq = (double)tmp.QuadPart / 1000.0;
#endif
}

void Timer::start()
{
#ifdef _WIN32
	QueryPerformanceCounter((LARGE_INTEGER*)& start_time);
#else//_WIN32
	gettimeofday(&start_time, 0);
#endif//_WIN32
}


void Timer::stop()
{
#ifdef _WIN32
	QueryPerformanceCounter((LARGE_INTEGER*)& finish_time);
#else//_WIN32
	gettimeofday(&finish_time, 0);
#endif//_WIN32
}

timer_dt Timer::get_time()
{
	timer_dt interval = 0.0f;

#ifdef _WIN32
	interval = (timer_dt)((double)(finish_time.QuadPart
		- start_time.QuadPart) / freq);
#else
	// time difference in milli-seconds
	interval = (timer_dt)(1000.0 * (finish_time.tv_sec - start_time.tv_sec)
		+ (0.001 * (finish_time.tv_usec - start_time.tv_usec)));
#endif//_WIN32

	return interval;
}


timer_dt Timer::stop_get()
{
	timer_dt interval;
	stop();
	interval = get_time();

	return interval;
}

// Stop the timer, get the time interval, then start the timer again.
timer_dt Timer::stop_get_start()
{
	timer_dt interval;
	stop();
	interval = get_time();
	start();

	return interval;
}

#endif//_TIMER_H_
