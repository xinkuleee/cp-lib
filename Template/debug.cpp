#include <bits/stdc++.h>

using namespace std;

template <class T, size_t size = tuple_size<T>::value>
string to_debug(T, string s = "") requires(not ranges::range<T>);

template<class T>
concept check = requires(T x, ostream &os) {
  os << x;
};

template<check T>
string to_debug(T x) {
  return static_cast<ostringstream>(ostringstream() << x).str();
}

string to_debug(ranges::range auto x, string s = "") requires(not is_same_v<decltype(x), string>) {
  for (auto xi : x) {
    s += ", " + to_debug(xi);
  }
  return "[ " + s.substr(s.empty() ? 0 : 2) + " ]";
}

template <class T, size_t size>
string to_debug(T x, string s) requires(not ranges::range<T>) {
  [&]<size_t... I>(index_sequence<I...>) {
    ((s += ", " + to_debug(get<I>(x))), ...);
  }(make_index_sequence<size>());
  return "{" + s.substr(s.empty() ? 0 : 2) + "}";
}

#define debug(...) [](auto... $){ ((cout << to_debug($) << " "), ...); cout << endl; }("[", #__VA_ARGS__, "]:", __VA_ARGS__)
