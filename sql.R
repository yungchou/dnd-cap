fid <- c(17998,29830,30980,23089)
aid <- c(1001,981,734,985)
t <- c('Post','Comment','Photo','Share')
rid <- c(345,1001,234,1001)

dt <- data.frame(fid,aid,t,rid)
library(sqldf)
sqldf('select * from dt')

sqldf('
select t, count(t), rid from dt group by t
      ')

sqldf('
select rid, count(t) as story from dt
where t in ("Post", "Photo", "Share") group by rid
      ')
sqldf('
select count(abc.t) as numComment from
(select t, count(rid) from dt
where t="Comment" and group by "Comment") as abc
')
