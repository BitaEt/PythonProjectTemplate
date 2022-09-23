create index idx_batter_counts on baseball.batter_counts(game_id);
create index idx_game on baseball.game(game_id);

create table baseball.results (batter numeric, annual_batting_average double PRECISION, historic_batting_average double precision, rolling_batting_average double PRECISION);

Question 1

annual
nsert into baseball.results (batter, annual_batting_average) select batter, batting_average from(
select btc.batter, sum(btc.hit)/sum(btc.atBat) as batting_average, extract(YEAR from gm.local_date) as year
from baseball.batter_counts btc
left join baseball.game gm
on btc.game_id = gm.game_id
group by (btc.batter), year
order by (btc.batter), year) as aa

total
insert into baseball.results (historic_batting_average) select batter from (select batter, sum(hit)/sum(atBat) as batting_average
from baseball.batter_counts
group by batter)
where batter = batter;


Question 2

with t1 as (select btc.batter, max(gm.local_date) as max from baseball.batter_counts btc
left join baseball.game gm
on btc.game_id = gm.game_id
group by btc.batter),
t2 as (select btc.batter, sum(btc.hit)/sum(btc.atBat) as batting_average, gm.local_date
from baseball.batter_counts btc
left join baseball.game gm
on btc.game_id = gm.game_id
group by btc.batter, btc.game_id)
select t2.batter , avg(t2.batting_average)  from t2
right join t1 on t2.batter = t1.batter
where t2.local_date > date_add(t1.max, INTERVAL -100 DAY)
group by t1.batter
