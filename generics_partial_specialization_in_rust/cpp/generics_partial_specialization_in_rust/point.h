#pragma once

template<typename T, size_t D>
struct Point;

template<typename T>
struct Point<T, 2> {
	T x;
	T y;
	Point(const T& x, const T& y) : x{ x }, y{ y } {}
};

template<typename T>
struct Point<T, 3> {
	T x;
	T y;
	T z;
	Point(const T& x, const T& y, const T& z) : x{ x }, y{ y }, z{ z } {}
};
