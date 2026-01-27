## Database Operations with Fastlite

### 1. Opening Files, Creating Tables, and Deleting Tables

```python
from fastlite import *
from dataclasses import dataclass
from fastcore.all import *
db = Database('db.sqlite')

@dataclass
class UserInfo:
    id: int
    username: str
    password_hash: str

userinfo = db.create(UserInfo)
db.close()
```

Whether opening an existing database or creating a new one, simply pass the filename to `Database`. Then, when creating tables, we use the more elegant `Dataclass` format, saving us from writing cumbersome SQL statements to define tables. `Fastlite` will automatically recognize `id` as the primary key.

After creating a table with `create`, we have two ways to examine the table's conversion/definition:

- `hl_md(userinfo.schema, 'sql')` outputs the corresponding SQL statement used to create the table.
- `userinfo.dataclass()` returns the `dataclass` corresponding to the table. This line of code will still work even if we opened a new database and didn't define `UserInfo` in the file.
- We can also view the source code of the dataclass: `hl_md(dataclass_src(userinfo.dataclass()), 'python')`

Another, more ordinary, method of creating a table is to do it in a single line.

```python
cats.create(id=int, name=str, weight=float, uid=int, pk='id')
```

If you want to delete a table, use `cats.drop()`; the data inside will be deleted.

### 2. Examining Table Contents

This refers to quickly understanding the main structure of an unfamiliar database. We can use the following methods.

Using `db.t` lists all table names in the database. `dt.*table_name*` directly fetches a specific table. You can use `'table_name' in dt` to check if a table exists in the database.

```python
dt = db.t
print(dt)
userinfo = dt.user_info
```

After getting a table, you can get all its columns via `userinfo.c`.

### 3. Inserting Data

```python
userinfo = db.t.user_info
userinfo.insert({'username': 'jason', 'password_hash': ''})
```

We can directly insert dictionaries, or we can insert objects in the form of a Dataclass, entirely based on personal coding preference.

### 4. Querying and Getting All Data

Fastlite's query API is relatively inelegant, so I prefer to write the corresponding SQL query statement directly.

```python
query = 'SELECT password_hash from user_info WHERE username = ?'
result = db.q(query, (username, ))
```

The format of the returned data is as I imagined: a list containing tuples.

If you want to see all the data in a table at once, you can use `cats(limit=10)`, where `cats` is a table object.

### 5. Updating Data and Deleting Data

```python
cats.update({'id':1, 'name': 'jason cat'})
```

The inserted dictionary needs to include the primary key so the database knows exactly which row to update!

The same principle applies to deleting data: we only need to provide the primary key of the row to be deleted.

```python
cats.delete(1)
```

That's it.