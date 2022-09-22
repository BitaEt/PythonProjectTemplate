create index idx_batter_counts on baseball.batter_counts(game_id);
create index idx_game on baseball.game(game_id);


Question 1

annual
with t1 as (select btc.team_id, sum(btc.hit)/sum(btc.atBat) as batting_average, extract(YEAR from gm.local_date) as year
from baseball.batter_counts btc
left join baseball.game gm
on btc.game_id = gm.game_id
group by (btc.team_id))
update table baseball.team_results tr inner join t1
on tr.game_id = t1.game_id
set tr.annual_batting_average = t1.batting_average
limit 10;

total
select btc.team_id, sum(btc.hit)/sum(btc.atBat) as batting_average
from baseball.batter_counts btc
left join baseball.game gm
on btc.game_id = gm.game_id
group by (btc.team_id)
limit 10;


Question 2

with t1 as (select btc.team_id, max(gm.local_date) as max from baseball.batter_counts btc
left join baseball.game gm
on btc.game_id = gm.game_id),
t2 as (select btc.team_id, sum(btc.hit)/sum(btc.atBat) as batting_average, gm.local_date
from baseball.batter_counts btc
left join baseball.game gm
on btc.game_id = gm.game_id
group by btc.team_id, btc.game_id)
select t1.team_id , avg(t2.batting_average)  from t1 
right join t2 on t1.team_id = t2.team_id
where t2.local_date > date_add(t1.max, INTERVAL -100 DAY)
group by t1.team_id