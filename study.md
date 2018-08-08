##Diagonal Traversal

Given a matrix of M x N elements (M rows, N columns), return all elements of the matrix in diagonal order as shown in the below image.

Example:
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output:  [1,2,4,7,5,3,6,8,9]
Explanation:

Note:
The total number of elements of the given matrix will not exceed 10,000.

```javascript
var findDiagonalOrder = function(matrix) {

    if (!matrix) return [];
    if (!matrix[0]) return matrix;

    let m = matrix.length, n = matrix[0].length;
    let result = [];

    let row = 0, col = 0, d = 1;


    for (let i = 0; i < m * n; i++) {
        result.push(matrix[row][col]);

        row -= d;
        col += d;

        //bottom
        if (row >= m) {
            row = m - 1;
            col += 2;
            d = -d;
        }

        //right
        if (col >= n) {
            col = n - 1;
            row += 2;
            d = -d;
        }
        //top
        if (row < 0)  {
            row = 0;
            d = -d;
        }
        //left
        if (col < 0)  {
            col = 0;
            d = -d;
        }
    }

    return result;

};
```

There are four cases corresponding to the four sides of the matrix. We continue in the diagonal directional until we are out of bounds and then we correct.

##Construct Binary Tree from Preorder and Inorder Traversal
Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7

```javascript
var buildTree = function(preorder, inorder) {

    if(!preorder.length) return null;

    let rootVal = preorder[0];
    let rootInd = inorder.indexOf(rootVal);

    let root = new TreeNode(rootVal);

    root.left = buildTree(preorder.slice(1, rootInd + 1), inorder.slice(0, rootInd));
    root.right = buildTree(preorder.slice(rootInd + 1), inorder.slice(rootInd + 1));

    return root;


};
```
##Valid Parentheses
```javascript
var isValid = function(s) {

    let stack = [];

    let open  = {
        '{': '}',
        '[': ']',
        '(': ')',
    }

    let closed  = new Set(['}', ']', ')']);



    for (let i = 0; i < s.length; i++) {

        if (open[s[i]]) {
            stack.push(s[i]);
        } else if (closed.has(s[i])) {
            let match = stack.pop();
            if (open[match] !== s[i]) return false;
        }
    }


    return stack.length ? false : true;
};
```
##Write a function that takes a string and returns the longest substring without repeating characters.
```javascript
var lengthOfLongestSubstring = function(s) {

    let current = new Set();
    let length = 0;
    let head = 0;
    let tail = 0;

    while (tail < s.length) {
        if (!current.has(s[tail])) {
            current.add(s[tail]);
            tail++;
            length = Math.max(current.size, length);
        } else {
            current.delete(s[head])
            head++;
        }

    }

    return length;


};
```

##Fibonacci (recursive and non-recursive)
```javascript
function fib(n, memo = {}) {

  if (n <= 1) return n;

  let fibMinusOne = memo[n - 1] || fib(n - 1, memo);
  let fibMinusTwo = memo[n - 2] || fib(n - 2, memo);
  let ans = fibMinusOne + fibMinusTwo;

  memo[n] = ans;

  return ans;
}

function bottomUpFib(n) {
  if (n <= 1) return n;

  let hold = [0, 1];

  for (let i = 0; i < (n - 1) / 2; i++) {
    hold[0] = hold[0] + hold[1];
    hold[1] = hold[0] + hold[1];
  }

  return n % 2 === 0 ? hold[0] : hold[1]
}
```

##level order traversal

```javascript
var levelOrder = function(root) {
    let levelOrd = [];
    let q = [root];
    let current;

    while (q.length) {
        let size = q.length;

        let level = [];
        for (let i = 0; i < size; i++) {
            current = q.shift();
            if (current.left) q.push(current.left);
            if (current.right) q.push(current.right);
            level.push(current.val)
        }
        levelOrd.push(level);
    }

    return levelOrd;
};

```

##merge two sorted lists

```javascript
function mergeTwoLists(l1, l2){
		if(l1 == null) return l2;
		if(l2 == null) return l1;
		if(l1.val < l2.val){
			l1.next = mergeTwoLists(l1.next, l2);
			return l1;
		} else{
			l2.next = mergeTwoLists(l1, l2.next);
			return l2;
		}
}
```
##qiucksort
```javascript
function quickSort(nums, lo = 0, hi = arr.length - 1) {

  if (lo >= hi) return;

  for (var i = lo, j = lo; j < hi; j++) {
        if (nums[j] <= nums[hi]) {
            swap(nums, i++, j);
        }
    }
    swap(nums, i, j);


    quickSort(nums, i + 1, hi);
    quickSort(nums, lo, i - 1);
};

function swap(nums, i, j) {
    [nums[i], nums[j]] = [nums[j], nums[i]]
}
```

##mergesort

```javascript
function mergeSort(arr) {
  if (arr.length === 1) return arr;

  let midPoint = Math.floor(arr.length/2);

  let leftSorted = mergeSort(arr.slice(0, midPoint));
  let rightSorted = mergeSort(arr.slice(midPoint, arr.length));

  return merge(leftSorted, rightSorted);
}

function merge(left, right) {
  let result = [];

  while (left.length && right.length){
    if (left[0] < right[0]) {
      result.push(left.shift());
    } else {
      result.push(right.shift());
    }
  }

  return left.length > 0 ? result.concat(left) : result.concat(right);
}
```

##Databases

#Denormalized vs. Normalized:
Normalized db’s are designed to minimize redundancy while denormailzed dbs are designed to optimize read time. Denormilzation is commonly used to create highly scalable systems. In a normalized db courses might have a foreign key for teachers (with no redundancy) while in a denormalized db we might store the teachers name in the courses table.

#Acid
Atomicity
Transactions are single logical units of work (read and write). ACID principles are followed to maintain consistency.

A - Atomicity (all or nothing)
The entire transaction takes place or nothing happens at all. Therefor, a transaction can be aborted, and changes made to the db are not visible, or a transaction can be commited in which case they are visible.

C - concistency - Consistency ensures that a transaction can only bring the database from one valid state to another, maintaining database invariants: any data written to the database must be valid according to all defined rules, including constraints, cascades, triggers, and any combination thereof. This prevents database corruption by an illegal transaction, but does not guarantee that a transaction is correct.

I - Isolation − In a database system where more than one transaction are being executed simultaneously and in parallel, the property of isolation states that all the transactions will be carried out and executed as if it is the only transaction in the system. No transaction will affect the existence of any other transaction.

D - Durability - Once transactions are committed they must be written to a persistent memory and be able to survive system failure.

##Systems design

#load balancing

#cacheing

##node
#•What makes Node different than traditional, non-blocking web servers? How do you scale a Node application?
Node is really fast. Its non-blocking which means its really good at reading and writing and these operations don't block the stack. The core v8 javascript engine is also very fast. It allows for server-side rendering which can increase key metrics like time to render and time to interact.

The native Node.js cluster module is the basic way to scale a Node app on a single machine. A single instance of Node.js runs in a single thread. To take advantage of multi-core systems, the user will sometimes want to launch a cluster of Node.js processes to handle the load.

##error first callback (convention in node)
As a convention, callbacks in Node take the error object as the first argument, which may be nice becuase if you have an error you want to handle it first.

##last cb
in functions like fs.readfile the callback is always last.

##streams
Uses events to pass data, communicate errors, and the end of an input. Data is passed by using buffer objects or strings. Streams are readable, writable, both and transformational.

##clusters
Node is single threaded, so in order to take advantage of all of the cpu's of a machine we use clusters. If we didnt use clusters and we were doing a lot of processing, the other cpu's would be mostly dormant. The process object will tell you how many cores and you usually see a for loop forking once for every core.

The master forks the child processes. Each child process calls listen() but the master opens the socket and sends the requests to the children (round robin) who are listening on the port. Events coordinate the master and children: fork, online, listening, disconnect, exit, and setup.

```javascript
if (cluster.isMaster) {
  console.log(`Master ${process.pid} is running`);

  // Fork workers.
  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  cluster.on('exit', (worker, code, signal) => {
    console.log(`worker ${worker.process.pid} died`);
  });
} else {
  // Workers can share any TCP connection
  // In this case it is an HTTP server
  http.createServer((req, res) => {
    res.writeHead(200);
    res.end('hello world\n');
  }).listen(8000);
```

##What do the terms “CPU bound” and “I/O bound” mean?
A program is CPU bound if it would go faster if the CPU were faster, i.e. it spends the majority of its time simply using the CPU (doing calculations). A program that computes new digits of π will typically be CPU-bound, it's just crunching numbers.

A program is I/O bound if it would gfaster if the I/O subsystem was faster. Which exact I/O system is meant can vary; I typically associate it with disk, but of course networking or communication in general is common too. A program that looks through a huge file for some data might become I/O bound, since the bottleneck is then the reading of the data from disk (actually, this example is perhaps kind of old-fashioned these days with hundreds of MB/s coming in from SSDs).

##tcp sockets



##web application architecture

##authentication

Authentication refers to the process of checking that an entity is who they claim to be, i.e. a login process. This is different from authorization, which is more about controlling what resources a client should and should not have access to.

Local authentication as a term is contrasted with foreign strategies (or provider strategies) that come from third-party applications like Facebook and Twitter. Simply put, a user can decide to set up an account "directly", by providing a username and password, or "indirectly", through some other web application that both of you trust.

Sessions in Node
Sessions are ways of keeping a user logged in. One we can do this is by using cookies. Cookies allow you to store a user’s information inside a file on their browser. The browser then sends that info back on every request instead of sending the username and password on every request.

You can set cookies in responses. For instance, the “Set-Cookie” header might set the cookie value to a string like “session=r@rdegges.com”. Just like that, the user’s browser will store and pass along a cookie the next time they visit your site. Your server can grab the email and pull their profile information from your database if needed.

Cookies have a duration, after which they are invalidated (or logged out) but they can be extended by setting an active duration which will increase the time based on interaction with the site. Client-session libraries will also ask for a secret which should be a high-entropy string that you choose to encrypt the cookies.

##•What is a JS closure?
A closure is the combination of a function bundled together (enclosed) with references to its surrounding state (the lexical environment). In other words, a closure gives you access to an outer function’s scope from an inner function. In JavaScript, closures are created every time a function is created, at function creation time.
Examples: callbacks, data privacy, partial application

##Load Balancing

Load balancers are used to increase application availability and responsiveness while also preventing any one application server from becoming a single point of failure. It does this by spreading a load of traffic amongst a cluster of servers. Additionally, if a server is not responding or is responding with a high rate of error, LB will stop sending requests to that server.

Web server: serves content to the web using http protocol.
Application server: hosts and exposes business logic and processes. (not limited to http)
Database server: hosts the database
Load balancers can go in between all of these servers. Load balancer can be a single point of failure, to overcome this a second load balancer can be connected to the first to form a cluster. Each LB monitors the health of the other and since both of them are equally capable of serving traffic and failure detection, in the event the main load balancer fails, the second load balancer takes over.

##Cacheing
Load balancing helps you scale horizontally across an ever-increasing number of servers, but caching will enable you to make vastly better use of the resources you already have, as well as making otherwise unattainable product requirements feasible.
Caches can exist at all levels of architecture but are normally placed near the front-end as to not tax downstream levels.
Cahces can exist in memory, on disk, or CDN. CDN’s are good for serving up large amounts of static media.

Invalidation:
 If the data is modified in the database, it should be invalidated in the cache, if not, this can cause inconsistent application behavior.
Write-through: Write to both cache and db (high latency for double write)
Write around: write to db (cache misses on writes)
Write back: write to cache (low latency, poential for data loss)

##single source of truth (redux)
In redux your state exists in a single object that can be accessed through the global store. This makes code easier to debug and introspect.

##What are “actions” in Redux?
Actions represent events in the lifecycle of an application. Every action is an object with a type and payload of data. They are the means of sending data from an application to the store.

##What is the role of reducers in Redux?
Reducers take in actions and return a new state. It is important that they are pure functions and dont have sideffects. They can be combined in larger applications.

##What is ‘Store’ in Redux?
The store is the holder of the application state which also provides other methods like dispatch actions and registering listeners.






