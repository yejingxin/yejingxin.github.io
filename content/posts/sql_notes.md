---
title: "SQL Notes"
date: 2022-03-25T15:59:48-07:00
---

### `FROM -> WHERE -> AS` Executing Order
 - Why the defined column name cannot be used in `WHERE` clause? Because the `col AS new_col` is executed after `WHERE`. The executing order in the follow SQL is `FROM -> WHERE -> AS`:
 ```
 SELECT col AS new_col
 FROM table
 WHERE col < 10
 ```
 ### `EXISTS` operator
 - The `EXISTS` operator is used to test for the existence of any record in a subquery, it returns `TRUE` if the subquery returns one or more rows. `EXISTS ()` The subquery within the bracket after `EXISTS` can be treated as a function `f(row, table) -> bool` to filter the row.
 ```
 SELECT row
 FROM table_a
 WHERE EXISTS f(row, table_b, ...)
 ```
 ### `NULL` logic operator
  - `NULL` logic operator: note in normal programming language, any logic operation only has two outputs `TRUE` and `FALSE`, `NULL` is treated as `False`, but this is different in SQL, the logic output has three results: `TRUE`, `FALSE`, and `NULL`
    - `a is NULL` and `a is not NULL` can be inferred by normal logic, which returns `True` or `False`
    - any comparison operator with `NULL` will be `NULL`, example like `a = NULL` is `NULL`, even in the case `a = NULL`
    - `TRUE or NULL` is `TRUE`, since the second expression won't get a chance to be evaluated.
    - `FALSE or NULL` is `NULL`
