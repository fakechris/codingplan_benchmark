"""
Coding 能力评测任务集

专注测试 LLM 的实际编码能力:
1. 代码生成 (Code Generation) - 根据需求写代码
2. Bug 修复 (Bug Fix) - 找出并修复代码中的 bug
3. 代码补全 (Code Completion) - 补全部分代码
4. 重构优化 (Refactoring) - 改进已有代码
5. 算法实现 (Algorithm) - 实现数据结构/算法
6. 代码审查 (Code Review) - 识别代码问题
7. 测试编写 (Test Writing) - 为给定代码编写测试

设计原则:
- 每个任务都有可客观验证的 checkpoints
- 陷阱任务用于检测量化/掺水模型
- 覆盖 Python/Go/TypeScript 多语言
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class TaskCategory(Enum):
    CODE_GEN = "code_gen"
    BUG_FIX = "bug_fix"
    CODE_COMPLETION = "code_completion"
    REFACTORING = "refactoring"
    ALGORITHM = "algorithm"
    CODE_REVIEW = "code_review"
    TEST_WRITING = "test_writing"
    SYSTEM_DESIGN = "system_design"


@dataclass
class Checkpoint:
    """评判检查点"""
    description: str
    keywords: list[str]
    weight: float = 1.0
    is_critical: bool = False


@dataclass
class Task:
    """一个评测任务"""
    id: str
    title: str
    difficulty: TaskDifficulty
    category: TaskCategory
    prompt: str
    system_prompt: str = ""
    checkpoints: list[Checkpoint] = field(default_factory=list)
    max_tokens: int = 4096
    reference_plan: str = ""
    trap_type: Optional[str] = None
    tags: list[str] = field(default_factory=list)


# ============================================================
# 系统提示词
# ============================================================

SYS_CODER = "You are an expert software engineer. Write clean, correct, production-ready code. Include comments for complex logic."

SYS_REVIEWER = "You are a senior code reviewer. Be thorough and precise. Point out bugs, security issues, performance problems, and style issues."

# ============================================================
# 任务定义
# ============================================================

TASKS: list[Task] = [

    # ================================================================
    # DEBUG 专用 (极简, 用于连通性测试)
    # ================================================================
    Task(
        id="D01",
        title="Hello World 连通性测试",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.CODE_GEN,
        system_prompt=SYS_CODER,
        prompt='Write a Python function `greet(name: str) -> str` that returns "Hello, {name}!". Include a docstring.',
        checkpoints=[
            Checkpoint("函数定义正确", ["def greet", "name", "str"], weight=1.0, is_critical=True),
            Checkpoint("返回格式正确", ["Hello", "return", "f\"", "format"], weight=1.0),
        ],
        max_tokens=512,
        tags=["debug", "trivial"],
    ),

    # ================================================================
    # EASY - 代码生成
    # ================================================================
    Task(
        id="E01",
        title="IPv4 地址验证",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.CODE_GEN,
        system_prompt=SYS_CODER,
        prompt="""Write a Python function `is_valid_ipv4(ip: str) -> bool` that checks whether a string is a valid IPv4 address.

Requirements:
- Four decimal numbers separated by dots
- Each number is 0-255
- No leading zeros (e.g., "01.01.01.01" is invalid)
- No extra whitespace
- Return False for any invalid input

Give ONLY the function implementation, no test code.""",
        checkpoints=[
            Checkpoint("函数签名正确", ["def is_valid_ipv4", "-> bool"], weight=1.0, is_critical=True),
            Checkpoint("split by dot", ["split", ".", "4"], weight=1.5, is_critical=True),
            Checkpoint("范围检查 0-255", ["255", "0", "int", "range"], weight=1.5, is_critical=True),
            Checkpoint("前导零检查", ["leading", "zero", "01", "len", "startswith", "lstrip"], weight=2.0, is_critical=True),
            Checkpoint("异常处理", ["try", "except", "ValueError", "False"], weight=1.0),
        ],
        reference_plan="Split by '.', check len==4, each part is numeric, no leading zeros, int value in [0,255].",
        tags=["code_gen", "validation", "python"],
    ),

    Task(
        id="E02",
        title="LRU Cache 实现",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.ALGORITHM,
        system_prompt=SYS_CODER,
        prompt="""Implement an LRU (Least Recently Used) Cache in Python.

```python
class LRUCache:
    def __init__(self, capacity: int):
        ...
    def get(self, key: int) -> int:
        \"\"\"Return value if key exists, else -1. Mark as recently used.\"\"\"
        ...
    def put(self, key: int, value: int) -> None:
        \"\"\"Insert or update. Evict LRU item if at capacity.\"\"\"
        ...
```

Requirements:
- Both get and put must be O(1) time complexity
- Give the complete class implementation""",
        checkpoints=[
            Checkpoint("使用 OrderedDict 或 dict+双向链表", ["OrderedDict", "dict", "链表", "linked", "Node"], weight=2.0, is_critical=True),
            Checkpoint("get 操作 O(1) + move_to_end", ["get", "move_to_end", "pop", "移到"], weight=1.5, is_critical=True),
            Checkpoint("put 容量检查 + 淘汰", ["capacity", "popitem", "evict", "del", "len"], weight=1.5, is_critical=True),
            Checkpoint("完整的 class 实现", ["class LRUCache", "__init__", "self"], weight=1.0),
        ],
        reference_plan="Use collections.OrderedDict. get: move_to_end + return. put: if exists update+move, else insert; if over capacity, popitem(last=False).",
        tags=["algorithm", "data_structure", "python"],
    ),

    Task(
        id="E03",
        title="JSON 解析器 - 扁平化",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.CODE_GEN,
        system_prompt=SYS_CODER,
        prompt="""Write a Python function to flatten a nested JSON/dict into a single-level dict with dot-notation keys.

```python
def flatten_json(obj: dict, prefix: str = '') -> dict:
    ...
```

Example:
```python
flatten_json({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
# => {"a": 1, "b.c": 2, "b.d.e": 3}

flatten_json({"x": [1, 2, 3]})
# => {"x.0": 1, "x.1": 2, "x.2": 3}
```

Handle: nested dicts, lists (index as key), and primitive values.""",
        checkpoints=[
            Checkpoint("递归实现", ["def flatten", "递归", "recursive", "flatten_json"], weight=1.0, is_critical=True),
            Checkpoint("dict 递归处理", ["isinstance", "dict", "items", "prefix"], weight=1.5, is_critical=True),
            Checkpoint("list 用索引作 key", ["list", "enumerate", "index", "str("], weight=1.5, is_critical=True),
            Checkpoint("dot 拼接", [".", "f\"{prefix}", "join", "concat", "+"], weight=1.0),
        ],
        reference_plan="Recursion. For dict: recurse with prefix.key. For list: recurse with prefix.index. Else: assign value.",
        tags=["code_gen", "json", "python"],
    ),

    # ================================================================
    # EASY - Bug 修复
    # ================================================================
    Task(
        id="E04",
        title="修复二分查找 Bug",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.BUG_FIX,
        system_prompt=SYS_CODER,
        prompt="""The following binary search implementation has bugs. Find ALL bugs and provide the corrected version.

```python
def binary_search(arr, target):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) / 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1
```

List each bug, explain why it's wrong, and give the fully corrected function.""",
        checkpoints=[
            Checkpoint("Bug1: 整数除法", ["//", "int", "integer division", "floor"], weight=2.0, is_critical=True),
            Checkpoint("Bug2: right 初始值", ["len(arr) - 1", "len(arr)", "right = len", "inclusive", "exclusive"], weight=1.5, is_critical=True),
            Checkpoint("Bug3: left=mid 死循环", ["mid + 1", "mid - 1", "infinite loop", "死循环"], weight=2.0, is_critical=True),
            Checkpoint("溢出防护 (可选)", ["overflow", "left + (right - left)", "溢出"], weight=0.5),
            Checkpoint("修复后完整代码", ["def binary_search", "return", "while"], weight=1.0),
        ],
        reference_plan="Bug1: / -> //. Bug2: right=len(arr)-1 with <=, or keep len(arr) with <. Bug3: left=mid+1, right=mid-1 (if inclusive).",
        tags=["bug_fix", "algorithm", "python"],
    ),

    # ================================================================
    # MEDIUM - 代码生成
    # ================================================================
    Task(
        id="M01",
        title="并发安全的连接池",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.CODE_GEN,
        system_prompt=SYS_CODER,
        prompt="""Implement a thread-safe connection pool in Python using asyncio.

```python
class AsyncConnectionPool:
    def __init__(self, max_size: int, factory):
        \"\"\"
        max_size: maximum number of connections
        factory: async callable that creates a new connection
        \"\"\"
        ...

    async def acquire(self, timeout: float = 10.0):
        \"\"\"Acquire a connection. Wait if pool is exhausted. Raise TimeoutError if timeout.\"\"\"
        ...

    async def release(self, conn):
        \"\"\"Return a connection to the pool.\"\"\"
        ...

    async def close(self):
        \"\"\"Close all connections.\"\"\"
        ...
```

Requirements:
- Use asyncio.Queue or asyncio.Semaphore
- Handle timeout properly
- Lazy initialization (create connections on demand)
- Track total created connections
- Give the complete implementation""",
        checkpoints=[
            Checkpoint("使用 asyncio 同步原语", ["asyncio.Queue", "asyncio.Semaphore", "asyncio.Lock", "asyncio.Event"], weight=2.0, is_critical=True),
            Checkpoint("超时处理", ["timeout", "asyncio.wait_for", "TimeoutError", "asyncio.timeout"], weight=1.5, is_critical=True),
            Checkpoint("懒初始化", ["factory", "create", "_current_size", "lazy"], weight=1.5),
            Checkpoint("release 归还连接", ["release", "put", "queue", "append"], weight=1.5, is_critical=True),
            Checkpoint("close 清理所有", ["close", "while", "all", "清理"], weight=1.0),
        ],
        reference_plan="asyncio.Queue for idle connections. Semaphore(max_size) to limit total. acquire: try get_nowait, else create new if under limit, else wait_for with timeout. release: put back to queue.",
        tags=["code_gen", "async", "concurrency", "python"],
    ),

    Task(
        id="M02",
        title="TypeScript 类型体操 - DeepPartial",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.CODE_GEN,
        system_prompt=SYS_CODER,
        prompt="""Implement the following TypeScript utility types:

1. `DeepPartial<T>` - Makes all properties (including nested) optional
2. `DeepReadonly<T>` - Makes all properties (including nested) readonly
3. `PickByType<T, U>` - Pick properties whose value type is U

Example usage:
```typescript
interface User {
  name: string;
  age: number;
  address: {
    street: string;
    city: string;
    geo: { lat: number; lng: number };
  };
  tags: string[];
}

type PartialUser = DeepPartial<User>;
// All fields including nested address.geo should be optional

type ReadonlyUser = DeepReadonly<User>;
// All fields including nested should be readonly

type StringProps = PickByType<User, string>;
// { name: string }
```

Give the complete type definitions.""",
        checkpoints=[
            Checkpoint("DeepPartial 递归定义", ["DeepPartial", "Partial", "keyof", "?:", "extends object"], weight=2.0, is_critical=True),
            Checkpoint("DeepReadonly 递归定义", ["DeepReadonly", "readonly", "Readonly", "keyof"], weight=2.0, is_critical=True),
            Checkpoint("PickByType 条件类型", ["PickByType", "extends", "as", "never", "infer"], weight=2.0, is_critical=True),
            Checkpoint("处理数组类型", ["Array", "[]", "ReadonlyArray", "infer"], weight=1.0),
            Checkpoint("mapped types 语法正确", ["[K in keyof", "T[K]", "mapped"], weight=1.0),
        ],
        reference_plan="DeepPartial: mapped type with conditional recursion. DeepReadonly: same with readonly modifier. PickByType: mapped type with key remapping via 'as'.",
        tags=["code_gen", "typescript", "types"],
    ),

    # ================================================================
    # MEDIUM - Bug 修复
    # ================================================================
    Task(
        id="M03",
        title="修复 Race Condition",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.BUG_FIX,
        system_prompt=SYS_CODER,
        prompt="""This Go code has race conditions and a goroutine leak. Find ALL bugs and fix them.

```go
package main

import (
    "fmt"
    "net/http"
    "time"
)

func fetchAll(urls []string) map[string]string {
    results := make(map[string]string)
    
    for _, url := range urls {
        go func() {
            resp, err := http.Get(url)
            if err != nil {
                results[url] = "error: " + err.Error()
                return
            }
            defer resp.Body.Close()
            results[url] = resp.Status
        }()
    }
    
    time.Sleep(5 * time.Second)
    return results
}

func main() {
    urls := []string{
        "https://example.com",
        "https://google.com",
        "https://github.com",
    }
    results := fetchAll(urls)
    for url, status := range results {
        fmt.Printf("%s: %s\\n", url, status)
    }
}
```

List each bug, explain the root cause, and provide the fully corrected code.""",
        checkpoints=[
            Checkpoint("Bug1: 闭包变量捕获", ["closure", "闭包", "capture", "url", "parameter", "参数", "shadow"], weight=2.0, is_critical=True),
            Checkpoint("Bug2: map 并发写不安全", ["map", "concurrent", "race", "sync.Mutex", "sync.Map", "channel"], weight=2.5, is_critical=True),
            Checkpoint("Bug3: sleep 替代同步", ["WaitGroup", "sync.WaitGroup", "channel", "sleep", "Done", "Wait"], weight=2.0, is_critical=True),
            Checkpoint("Bug4: 无超时控制", ["context", "timeout", "http.Client", "Timeout", "cancel"], weight=1.0),
            Checkpoint("修复后完整代码", ["func fetchAll", "sync.WaitGroup", "go func"], weight=1.0),
        ],
        reference_plan="Fix1: pass url as goroutine param. Fix2: use sync.Mutex or channel to collect results. Fix3: use sync.WaitGroup instead of sleep. Fix4: use http.Client with timeout.",
        trap_type="edge_case",
        tags=["bug_fix", "concurrency", "go", "trap"],
    ),

    # ================================================================
    # MEDIUM - 代码审查
    # ================================================================
    Task(
        id="M04",
        title="代码审查: Node.js API",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.CODE_REVIEW,
        system_prompt=SYS_REVIEWER,
        prompt="""Review the following Express.js API endpoint for bugs, security issues, and performance problems.

```javascript
const express = require('express');
const mysql = require('mysql');
const app = express();

const db = mysql.createConnection({
  host: 'localhost', user: 'root', password: 'admin123', database: 'users'
});

app.get('/user', (req, res) => {
  const userId = req.query.id;
  const query = `SELECT * FROM users WHERE id = ${userId}`;
  
  db.query(query, (err, results) => {
    if (err) {
      res.status(500).send(err.message);
      return;
    }
    
    const user = results[0];
    res.json({
      id: user.id,
      name: user.name,
      email: user.email,
      password: user.password,
      ssn: user.ssn,
    });
  });
});

app.post('/user', (req, res) => {
  const { name, email, password } = req.body;
  const query = `INSERT INTO users (name, email, password) VALUES ('${name}', '${email}', '${password}')`;
  
  db.query(query, (err, result) => {
    if (err) throw err;
    res.json({ id: result.insertId });
  });
});

app.listen(3000);
```

For each issue found:
1. Describe the problem
2. Explain the severity (Critical/High/Medium/Low)
3. Give the fix

Be thorough - there are at least 6 issues.""",
        checkpoints=[
            Checkpoint("SQL 注入漏洞", ["SQL injection", "SQL 注入", "parameterized", "prepared", "placeholder", "?"], weight=3.0, is_critical=True),
            Checkpoint("密码泄露", ["password", "密码", "敏感", "sensitive", "expose", "泄露", "hash"], weight=2.5, is_critical=True),
            Checkpoint("SSN 泄露", ["ssn", "SSN", "社会安全号", "PII", "敏感信息"], weight=1.5),
            Checkpoint("明文密码存储", ["hash", "bcrypt", "加密", "明文", "plaintext"], weight=2.0, is_critical=True),
            Checkpoint("错误处理: throw err", ["throw", "crash", "unhandled", "process exit", "try-catch"], weight=1.5),
            Checkpoint("缺少输入验证", ["validation", "验证", "sanitize", "check", "empty", "null"], weight=1.0),
            Checkpoint("错误信息泄露", ["error message", "err.message", "stack trace", "错误信息", "泄露"], weight=1.0),
            Checkpoint("缺少 body parser / rate limit", ["body-parser", "json()", "rate limit", "middleware"], weight=0.5),
        ],
        reference_plan="Issues: 1) SQL injection in both routes, 2) password/ssn in response, 3) passwords stored in plaintext, 4) throw err will crash server, 5) no input validation, 6) error message leak, 7) no body parser, 8) hardcoded credentials.",
        tags=["code_review", "security", "nodejs"],
    ),

    # ================================================================
    # MEDIUM - 测试编写
    # ================================================================
    Task(
        id="M05",
        title="编写单元测试",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.TEST_WRITING,
        system_prompt=SYS_CODER,
        prompt="""Write comprehensive pytest tests for this function:

```python
from datetime import datetime, timedelta
from typing import Optional

def parse_relative_time(text: str) -> Optional[datetime]:
    \"\"\"Parse relative time strings into datetime objects.

    Supported formats:
    - "now" -> current time
    - "5m ago" / "5 minutes ago" -> 5 minutes before now
    - "2h ago" / "2 hours ago"
    - "3d ago" / "3 days ago"
    - "1w ago" / "1 week ago"
    - "in 5m" / "in 5 minutes" -> 5 minutes from now
    - "in 2h" / "in 2 hours"
    - Combinations are NOT supported
    - Returns None for invalid input
    \"\"\"
    ...
```

Requirements:
- Use pytest with `freezegun` or manual mocking for time
- Cover all supported formats
- Test edge cases: invalid input, zero values, large numbers
- Test both "ago" and "in" directions
- At least 15 test cases
- Use parametrize where appropriate""",
        checkpoints=[
            Checkpoint("pytest parametrize 使用", ["@pytest.mark.parametrize", "parametrize", "params"], weight=2.0, is_critical=True),
            Checkpoint("时间 mock/freeze", ["freezegun", "freeze_time", "mock", "patch", "monkeypatch", "datetime"], weight=1.5),
            Checkpoint("覆盖所有时间单位", ["minutes", "hours", "days", "week", "5m", "2h", "3d", "1w"], weight=2.0, is_critical=True),
            Checkpoint("测试 ago 和 in 两个方向", ["ago", "in 5", "in 2h", "future", "past"], weight=1.5, is_critical=True),
            Checkpoint("测试无效输入", ["None", "invalid", "empty", '""', "garbage", "error"], weight=1.5),
            Checkpoint("边界: 零值/大数", ["0", "zero", "large", "9999", "overflow"], weight=1.0),
            Checkpoint("测试数量 >= 15", ["def test_", "test_"], weight=1.0),
        ],
        reference_plan="Use pytest.mark.parametrize for systematic coverage. Mock datetime.now(). Test each unit (m/h/d/w), both directions, edge cases (0, large, invalid format, empty string, None).",
        tags=["test_writing", "pytest", "python"],
    ),

    # ================================================================
    # HARD - 算法实现
    # ================================================================
    Task(
        id="H01",
        title="实现跳表 (Skip List)",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.ALGORITHM,
        system_prompt=SYS_CODER,
        prompt="""Implement a Skip List in Python with the following interface:

```python
class SkipList:
    def __init__(self, max_level: int = 16, p: float = 0.5):
        ...
    
    def search(self, target: int) -> bool:
        \"\"\"Return True if target exists.\"\"\"
        ...
    
    def insert(self, val: int) -> None:
        \"\"\"Insert val. Allow duplicates.\"\"\"
        ...
    
    def delete(self, val: int) -> bool:
        \"\"\"Delete one occurrence. Return True if found.\"\"\"
        ...
    
    def __str__(self) -> str:
        \"\"\"Visual representation showing all levels.\"\"\"
        ...
```

Requirements:
- Randomized level generation with probability p
- O(log n) average time for search/insert/delete
- Proper forward pointer management
- Include the Node class definition
- Give the COMPLETE implementation""",
        checkpoints=[
            Checkpoint("Node 定义 with forward pointers", ["class Node", "forward", "level", "key", "val"], weight=2.0, is_critical=True),
            Checkpoint("随机层级生成", ["random", "level", "p", "0.5", "max_level"], weight=1.5, is_critical=True),
            Checkpoint("search 逐层下降", ["search", "for i in range", "level", "forward", "while"], weight=2.0, is_critical=True),
            Checkpoint("insert 更新 update 数组", ["update", "insert", "forward", "level", "new_node"], weight=2.0, is_critical=True),
            Checkpoint("delete 正确解链", ["delete", "update", "forward", "None", "level"], weight=2.0, is_critical=True),
            Checkpoint("header/sentinel 节点", ["header", "head", "sentinel", "dummy", "float('-inf')"], weight=1.0),
        ],
        reference_plan="Node with forward[]. search: start from top level, go right while forward<target, then go down. insert: collect update[] path, random level, splice new node. delete: collect update[], remove node from each level.",
        tags=["algorithm", "data_structure", "python"],
    ),

    Task(
        id="H02",
        title="实现简易正则引擎",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.ALGORITHM,
        system_prompt=SYS_CODER,
        prompt="""Implement a simple regex engine that supports:
- `.` matches any single character
- `*` matches zero or more of the previous element
- `+` matches one or more of the previous element
- `?` matches zero or one of the previous element
- `^` matches start of string
- `$` matches end of string
- Character classes: `[abc]`, `[a-z]`, `[^abc]`

```python
def regex_match(pattern: str, text: str) -> bool:
    \"\"\"Return True if pattern matches the ENTIRE text.\"\"\"
    ...
```

Requirements:
- First compile pattern to NFA, then simulate
- Handle all edge cases
- Give complete implementation with the NFA state machine""",
        checkpoints=[
            Checkpoint("NFA 状态定义", ["State", "NFA", "state", "class", "transition", "accept"], weight=2.0, is_critical=True),
            Checkpoint("pattern 解析/编译", ["parse", "compile", "token", "ast", "postfix"], weight=2.0, is_critical=True),
            Checkpoint("* 零或多次匹配", ["star", "*", "zero or more", "epsilon", "closure"], weight=1.5, is_critical=True),
            Checkpoint("+ 和 ? 支持", ["+", "?", "one or more", "zero or one", "optional"], weight=1.5),
            Checkpoint("字符类 [] 解析", ["[", "]", "class", "range", "negate", "^"], weight=1.5),
            Checkpoint("NFA 模拟/执行", ["simulate", "current_states", "next_states", "epsilon", "closure", "step"], weight=2.0, is_critical=True),
        ],
        reference_plan="Thompson's construction: parse pattern -> NFA via State objects with epsilon transitions. Simulate NFA: maintain set of current states, advance on each char, compute epsilon closures.",
        tags=["algorithm", "regex", "python", "complex"],
    ),

    # ================================================================
    # HARD - 重构
    # ================================================================
    Task(
        id="H03",
        title="重构: 消除 God Object",
        difficulty=TaskDifficulty.HARD,
        category=TaskCategory.REFACTORING,
        system_prompt=SYS_CODER,
        prompt="""Refactor this "God Object" into a clean, well-structured design. Explain each refactoring step.

```python
class OrderSystem:
    def __init__(self, db_url):
        self.db = Database(db_url)
        self.email_host = "smtp.example.com"
        self.tax_rate = 0.08
        self.discount_rules = {}
        self.inventory = {}

    def create_order(self, user_id, items):
        # Validate user
        user = self.db.query(f"SELECT * FROM users WHERE id={user_id}")
        if not user: raise ValueError("User not found")
        if user['banned']: raise ValueError("User is banned")

        # Check inventory
        for item in items:
            stock = self.inventory.get(item['product_id'], 0)
            if stock < item['quantity']:
                raise ValueError(f"Insufficient stock for {item['product_id']}")

        # Calculate price
        total = 0
        for item in items:
            price = self.db.query(f"SELECT price FROM products WHERE id={item['product_id']}")
            subtotal = price * item['quantity']
            # Apply discount
            if item['product_id'] in self.discount_rules:
                discount = self.discount_rules[item['product_id']]
                subtotal *= (1 - discount)
            total += subtotal
        # Add tax
        total *= (1 + self.tax_rate)

        # Create order in DB
        order_id = self.db.execute(
            f"INSERT INTO orders (user_id, total) VALUES ({user_id}, {total})")

        # Update inventory
        for item in items:
            self.inventory[item['product_id']] -= item['quantity']

        # Send confirmation email
        import smtplib
        server = smtplib.SMTP(self.email_host)
        server.sendmail("noreply@example.com", user['email'],
                       f"Order {order_id} confirmed. Total: ${total:.2f}")
        server.quit()

        # Log
        print(f"Order {order_id} created for user {user_id}, total: {total}")
        return order_id

    def cancel_order(self, order_id):
        order = self.db.query(f"SELECT * FROM orders WHERE id={order_id}")
        if not order: raise ValueError("Order not found")
        self.db.execute(f"UPDATE orders SET status='cancelled' WHERE id={order_id}")
        # Restore inventory
        items = self.db.query(f"SELECT * FROM order_items WHERE order_id={order_id}")
        for item in items:
            self.inventory[item['product_id']] += item['quantity']
        # Send cancellation email
        user = self.db.query(f"SELECT * FROM users WHERE id={order['user_id']}")
        import smtplib
        server = smtplib.SMTP(self.email_host)
        server.sendmail("noreply@example.com", user['email'],
                       f"Order {order_id} cancelled")
        server.quit()

    def generate_report(self, start_date, end_date):
        orders = self.db.query(
            f"SELECT * FROM orders WHERE date BETWEEN '{start_date}' AND '{end_date}'")
        total_revenue = sum(o['total'] for o in orders)
        total_orders = len(orders)
        avg_order = total_revenue / total_orders if total_orders else 0
        return {"revenue": total_revenue, "orders": total_orders, "avg": avg_order}
```

Provide:
1. Identified problems (at least 5 specific issues)
2. Refactored code with clean class separation
3. Use SOLID principles and design patterns""",
        checkpoints=[
            Checkpoint("识别 SRP 违反", ["SRP", "Single Responsibility", "单一职责", "too many", "God"], weight=2.0, is_critical=True),
            Checkpoint("拆分为独立服务/类", ["OrderService", "NotificationService", "InventoryService", "PricingService", "UserService", "Repository"], weight=2.5, is_critical=True),
            Checkpoint("依赖注入", ["inject", "注入", "__init__", "constructor", "DI", "interface"], weight=2.0, is_critical=True),
            Checkpoint("SQL 注入问题", ["SQL injection", "parameterized", "ORM", "prepared"], weight=1.5),
            Checkpoint("事务一致性", ["transaction", "事务", "rollback", "atomic"], weight=1.0),
            Checkpoint("设计模式应用", ["Repository", "Strategy", "Observer", "Factory", "Service", "pattern"], weight=1.5),
        ],
        reference_plan="Split into: OrderService (orchestration), InventoryService, PricingService (with DiscountStrategy), NotificationService, OrderRepository. Use DI. Fix SQL injection. Add transactions.",
        tags=["refactoring", "solid", "python", "complex"],
    ),

    # ================================================================
    # EXPERT - 复杂编码 (陷阱题)
    # ================================================================
    Task(
        id="X01",
        title="Go 并发 Bug 修复",
        difficulty=TaskDifficulty.EXPERT,
        category=TaskCategory.BUG_FIX,
        system_prompt=SYS_CODER,
        prompt="""This Go connection pool has at least 3 subtle concurrency bugs. Find ALL of them.

```go
type Pool struct {
    mu       sync.Mutex
    conns    []*Conn
    maxSize  int
    factory  func() (*Conn, error)
    waiters  []chan *Conn
}

func (p *Pool) Get(ctx context.Context) (*Conn, error) {
    p.mu.Lock()
    if len(p.conns) > 0 {
        conn := p.conns[len(p.conns)-1]
        p.conns = p.conns[:len(p.conns)-1]
        p.mu.Unlock()
        return conn, nil
    }

    ch := make(chan *Conn)
    p.waiters = append(p.waiters, ch)
    p.mu.Unlock()

    select {
    case conn := <-ch:
        return conn, nil
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}

func (p *Pool) Put(conn *Conn) {
    p.mu.Lock()
    if len(p.waiters) > 0 {
        ch := p.waiters[0]
        p.waiters = p.waiters[1:]
        p.mu.Unlock()
        ch <- conn
        return
    }
    p.conns = append(p.conns, conn)
    p.mu.Unlock()
}
```

For each bug:
1. Describe the exact bug
2. Explain how it can be triggered (specific goroutine interleaving)
3. Give the fix with corrected code
4. Design a test that would expose the bug""",
        checkpoints=[
            Checkpoint("Bug1: ctx 取消后 waiter 泄漏", ["context", "cancel", "泄漏", "leak", "waiter", "remove"], weight=3.0, is_critical=True),
            Checkpoint("Bug2: Put 阻塞在 ch<-conn (对端已取消)", ["Put", "block", "阻塞", "deadlock", "send", "ch <-"], weight=3.0, is_critical=True),
            Checkpoint("Bug3: 连接泄漏 (ctx取消时连接丢失)", ["connection leak", "连接泄漏", "conn", "lost", "丢失"], weight=2.0, is_critical=True),
            Checkpoint("修复: non-blocking send", ["select", "default", "non-blocking", "非阻塞"], weight=2.0),
            Checkpoint("修复: 清理 waiter 列表", ["clean", "remove", "清理", "delete"], weight=1.5),
            Checkpoint("测试方案", ["test", "goroutine", "race", "-race", "context.WithTimeout"], weight=1.0),
        ],
        reference_plan="Bug1: ctx cancel leaves ch in waiters list. Bug2: Put sends on ch but receiver already gone - permanent block. Bug3: when ctx cancels after Put started sending, conn is lost. Fix: non-blocking select in Put, clean up waiter on ctx cancel, recover conn if send fails.",
        trap_type="edge_case",
        tags=["bug_fix", "concurrency", "go", "expert", "trap"],
    ),

    Task(
        id="X02",
        title="实现 Promise.allSettled + 重试",
        difficulty=TaskDifficulty.EXPERT,
        category=TaskCategory.CODE_GEN,
        system_prompt=SYS_CODER,
        prompt="""Implement a robust async task executor in Python:

```python
@dataclass
class TaskResult:
    task_id: str
    status: str  # "fulfilled" | "rejected"
    value: Any   # result if fulfilled
    error: str   # error message if rejected
    attempts: int
    duration: float

async def execute_with_retry(
    tasks: dict[str, Callable[[], Awaitable[Any]]],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout_per_task: float = 30.0,
    max_concurrency: int = 10,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> list[TaskResult]:
    \"\"\"
    Execute multiple async tasks with:
    - Concurrent execution with concurrency limit
    - Per-task timeout
    - Exponential backoff retry
    - Progress callback
    - Never throws - all errors captured in TaskResult
    \"\"\"
```

Requirements:
- Use asyncio.Semaphore for concurrency control
- Exponential backoff: delay * (backoff_factor ** attempt)
- Jitter on retry delay (random 0-25%)
- Per-task timeout using asyncio.wait_for
- Never raise exceptions - capture in TaskResult
- Track timing per task
- Give the COMPLETE working implementation""",
        checkpoints=[
            Checkpoint("Semaphore 并发控制", ["Semaphore", "semaphore", "max_concurrency", "asyncio.Semaphore"], weight=2.0, is_critical=True),
            Checkpoint("指数退避 + jitter", ["backoff", "exponential", "jitter", "random", "delay * ", "** attempt"], weight=2.0, is_critical=True),
            Checkpoint("asyncio.wait_for 超时", ["wait_for", "timeout", "asyncio.timeout", "TimeoutError"], weight=1.5, is_critical=True),
            Checkpoint("错误捕获不抛出", ["try", "except", "Exception", "rejected", "error"], weight=1.5, is_critical=True),
            Checkpoint("重试逻辑", ["retry", "attempts", "max_retries", "for i in range", "while"], weight=2.0, is_critical=True),
            Checkpoint("asyncio.gather 并行", ["gather", "create_task", "asyncio.gather", "tasks"], weight=1.0),
            Checkpoint("计时", ["time.time", "time.monotonic", "duration", "start"], weight=0.5),
        ],
        reference_plan="For each task: semaphore.acquire, retry loop with exponential backoff, wait_for timeout, capture all exceptions. gather all wrapped tasks. Return TaskResult list.",
        trap_type="complexity",
        tags=["code_gen", "async", "python", "expert", "trap"],
    ),

    Task(
        id="X03",
        title="实现简易 Git diff 算法",
        difficulty=TaskDifficulty.EXPERT,
        category=TaskCategory.ALGORITHM,
        system_prompt=SYS_CODER,
        prompt="""Implement the Myers diff algorithm to compute the shortest edit script between two sequences.

```python
@dataclass
class DiffOp:
    op: str      # "equal", "insert", "delete"
    old_line: int  # line number in old (-1 if insert)
    new_line: int  # line number in new (-1 if delete)
    text: str

def myers_diff(old: list[str], new: list[str]) -> list[DiffOp]:
    \"\"\"
    Compute minimal diff between old and new using Myers algorithm.
    Returns a list of DiffOps.
    \"\"\"
    ...

def format_unified_diff(ops: list[DiffOp], context: int = 3) -> str:
    \"\"\"Format diff ops as unified diff format (like git diff).\"\"\"
    ...
```

Requirements:
- Implement the actual Myers algorithm (not just LCS)
- O((M+N)*D) time where D is the edit distance
- Output both the diff operations and unified diff format
- Handle edge cases: empty inputs, identical inputs
- Give the COMPLETE implementation""",
        checkpoints=[
            Checkpoint("Myers D-loop 算法框架", ["for d in range", "D", "k", "diagonal", "furthest", "V"], weight=3.0, is_critical=True),
            Checkpoint("前向路径追踪", ["trace", "path", "backtrack", "snake", "V[k]"], weight=2.0, is_critical=True),
            Checkpoint("正确的 insert/delete 判断", ["insert", "delete", "equal", "DiffOp"], weight=1.5, is_critical=True),
            Checkpoint("unified diff 格式", ["@@", "---", "+++", "-", "+", "context", "hunk"], weight=1.5),
            Checkpoint("边界处理", ["empty", "len(old) == 0", "len(new) == 0", "identical"], weight=1.0),
            Checkpoint("O((M+N)*D) 复杂度", ["O(", "M+N", "edit distance", "shortest"], weight=1.0),
        ],
        reference_plan="Myers algorithm: D-loop from 0 to M+N, for each k in range(-D, D+1, 2), track furthest reaching paths on each diagonal. Backtrack to recover edit script.",
        trap_type="complexity",
        tags=["algorithm", "diff", "python", "expert", "trap"],
    ),

    # ================================================================
    # 速度/吞吐 专项测试
    # ================================================================
    Task(
        id="S01",
        title="简单函数生成 (速度基准)",
        difficulty=TaskDifficulty.EASY,
        category=TaskCategory.CODE_GEN,
        system_prompt="You are a concise coder. Give only code, no explanation.",
        prompt="Write a Python function `fizzbuzz(n: int) -> list[str]` that returns FizzBuzz results from 1 to n.",
        checkpoints=[
            Checkpoint("正确实现", ["def fizzbuzz", "Fizz", "Buzz", "FizzBuzz", "%"], weight=1.0),
        ],
        max_tokens=512,
        tags=["speed_test", "baseline"],
    ),

    Task(
        id="S02",
        title="长代码生成 (吞吐基准)",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.CODE_GEN,
        system_prompt=SYS_CODER,
        prompt="""Implement a complete REST API framework in Python (like a mini Flask/FastAPI) with:
1. Route registration with decorators (@app.get, @app.post, etc.)
2. Path parameter extraction (/users/{user_id})
3. Query parameter parsing
4. JSON request/response handling
5. Middleware support (before/after request hooks)
6. Error handling with custom exception classes
7. A simple test server using asyncio

Give the COMPLETE implementation with all features working. Include example usage.""",
        checkpoints=[
            Checkpoint("路由装饰器", ["@app.get", "@app.post", "route", "decorator", "def get"], weight=1.0),
            Checkpoint("路径参数提取", ["{", "}", "path", "param", "regex", "match"], weight=1.0),
            Checkpoint("中间件", ["middleware", "before", "after", "hook", "next"], weight=1.0),
        ],
        max_tokens=8192,
        tags=["throughput_test", "long_output"],
    ),

    # ================================================================
    # 一致性测试 (量化掺水检测)
    # ================================================================
    Task(
        id="C01",
        title="复杂度分析 (一致性测试)",
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.CODE_REVIEW,
        system_prompt="You are a precise algorithm analyst. Show your work step by step.",
        prompt="""Analyze the EXACT time complexity of this function. Show the full derivation.

```python
def solve(n):
    if n <= 1:
        return 1
    count = 0
    for i in range(n):
        for j in range(i, n):
            count += 1
    return count + solve(n // 3) + solve(n // 3)
```

Questions:
1. Write the recurrence relation T(n)
2. How many times is solve(n//3) called at each level?
3. What is the total work at recursion depth k?
4. Solve the recurrence - give the tight Big-O bound
5. Would memoization help? Why or why not?""",
        checkpoints=[
            Checkpoint("正确的递推关系", ["T(n)", "T(n/3)", "n^2", "2T(n/3)"], weight=2.0, is_critical=True),
            Checkpoint("循环部分 O(n^2)", ["n^2", "n*(n+1)/2", "n²", "quadratic"], weight=2.0, is_critical=True),
            Checkpoint("递归树分析", ["tree", "2^k", "level", "depth", "log", "层"], weight=2.0, is_critical=True),
            Checkpoint("最终复杂度正确", ["O(n^2)", "O(n²)", "n^2 dominates", "主定理"], weight=3.0, is_critical=True),
            Checkpoint("memoization 分析", ["memo", "缓存", "memoization", "重复计算", "不会"], weight=1.0),
        ],
        max_tokens=2048,
        trap_type="edge_case",
        reference_plan="T(n) = 2T(n/3) + O(n^2). By Master theorem: a=2, b=3, f(n)=n^2. log_3(2)≈0.63 < 2, so case 3 applies: T(n) = O(n^2).",
        tags=["consistency_test", "complexity_analysis", "trap"],
    ),
]


def get_tasks_by_difficulty(difficulty: Optional[TaskDifficulty] = None) -> list[Task]:
    if difficulty is None:
        return TASKS
    return [t for t in TASKS if t.difficulty == difficulty]


def get_tasks_by_tag(tag: str) -> list[Task]:
    return [t for t in TASKS if tag in t.tags]


def get_speed_tasks() -> list[Task]:
    return get_tasks_by_tag("speed_test")


def get_throughput_tasks() -> list[Task]:
    return get_tasks_by_tag("throughput_test")


def get_consistency_tasks() -> list[Task]:
    return get_tasks_by_tag("consistency_test")


def get_trap_tasks() -> list[Task]:
    return [t for t in TASKS if t.trap_type is not None]


def get_debug_tasks() -> list[Task]:
    return get_tasks_by_tag("debug")
