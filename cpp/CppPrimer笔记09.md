|      操作       |    作用    |  vector  |   deque    | list | forward_list | array |  string  |
| :-------------: | :--------: | :------: | :--------: | :--: | :----------: | :---: | :------: |
| `before_begin`  | 首前迭代器 |    —     |     —      |  —   |              |   —   |    —     |
|  **添加元素**   |            |          |            |      |              |       |          |
|   `push_back`   |            |          |            |      |      —       |   —   |          |
| `emplace_back`  |            |          |            |      |              |   —   |          |
|  `push_front`   |            |    —     |            |      |              |   —   |    —     |
| `emplace_front` |            |          |            |      |              |   —   |          |
|    `insert`     |            | 非尾很慢 | 非头尾很慢 |      |              |   —   | 非尾很慢 |
|    `emplace`    |            |          |            |      |              |   —   |          |
| `insert_after`  |            |    —     |     —      |  —   |              |   —   |    —     |
| `emplace_after` |            |    —     |     —      |  —   |              |   —   |    —     |
|  **访问元素**   |            |          |            |      |              |       |          |
|     `front`     |            |          |            |      |              |       |          |
|     `back`      |            |          |            |      |      —       |       |          |
|      `[n]`      |            |          |            |  —   |      —       |       |          |
|    `cat(n)`     |            |          |            |  —   |      —       |       |          |
|  **删除元素**   |            |          |            |      |              |       |          |
|   `pop_back`    |            |          |            |      |      —       |   —   |          |
|   `pop_front`   |            |    —     |            |      |              |   —   |    —     |
|     `erase`     |            |          |            |      |              |   —   |          |
|     `clear`     |            |          |            |      |              |   —   |          |
|  `erase_after`  |            |    —     |     —      |  —   |              |   —   |    —     |
|  **大小管理**   |            |          |            |      |              |       |          |
|    `resize`     |            |          |            |      |              |   —   |          |
|   `capacity`    |            |          |            |  —   |      —       |   —   |          |
| `shrink_to_fit` |            |          |            |      |              |       |          |
|   `capacity`    |            |          |     —      |  —   |      —       |   —   |          |
|    `reserve`    |            |          |     —      |  —   |      —       |   —   |          |



##### 6 适配器

|     适配器     |                  需要操作                  |      可用容器       | 默认容器 |
| :------------: | :----------------------------------------: | :-----------------: | :------: |
|     stack      |      `push_back`, `pop_back`, `back`       | vector, deque, list |  deque   |
|     queue      | `back`, `push_back`, `front`, `push_front` | vector,list, deque  |  deque   |
| Priority_queue | `front`, `push_back`, `pop_back`, 随机访问 |    vector, deque    |  vector  |

