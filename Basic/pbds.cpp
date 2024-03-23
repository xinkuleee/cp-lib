#include <bits/extc++.h>
using namespace __gnu_cxx;
using namespace __gnu_pbds;

#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
#include<ext/pb_ds/hash_policy.hpp>
#include<ext/pb_ds/trie_policy.hpp>
#include<ext/pb_ds/priority_queue.hpp>

pairing_heap_tag：配对堆
thin_heap_tag：斐波那契堆
binomial_heap_tag：二项堆
binary_heap_tag：二叉堆

__gnu_pbds::priority_queue<PII, greater<PII>, pairing_heap_tag> q;
__gnu_pbds::priority_queue<PII, greater<PII>, pairing_heap_tag>::point_iterator its[N];

its[v] = q.push({dis[v], v});
q.modify(its[v], {dis[v], v});

可以将两个优先队列中的元素合并（无任何约束）
使用方法为a.join(b)
此时优先队列b内所有元素就被合并进优先队列a中，且优先队列b被清空

cc_hash_table<string, int> mp1拉链法
gp_hash_table<string, int> mp2查探法
