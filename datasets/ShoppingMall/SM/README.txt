Dataset1: Visit y8 then y7 then y5 not visit certain places.
	Spurious: Only Time &  Only Visiting forbidden places & Mix
	Labels: 0-1000 Good 1000-1600 Bad 1600-1750 Good slow 1750-1900 Good fast 1900-1950 Bad fast 1950-2000 Bad slow
		0-150 Good 150-200 Bad 200-225 Good Slow 225-250 Good Fast 250-275 Bad Fast 275-300 Bad Slow


Dataset2: Do a certain path in some order
	Spurious:Do the same path in a different order & Time
	Labels: 0-1000 Good 1000-1600 Bad 1600-1750 Good slow 1750-1900 Good fast 1900-1950 Bad fast 1950-2000 Bad slow
		0-150 Good 150-200 Bad 200-225 Good Slow 225-250 Good Fast 250-275 Bad Fast 275-300 Bad Slow

Dataset3: Visit 4 places at the same time
	Spurious: Visit those 4 places but not at the same time
	Labels: 0-1000 Good 1000-1600 Bad 1600-1750 Good slow 1750-1900 Good fast 1900-1950 Bad fast 1950-2000 Bad slow
		0-150 Good 150-200 Bad 200-225 Good Slow 225-250 Good Fast 250-275 Bad Fast 275-300 Bad Slow


Dataset4: Branching path being able to choose 2 paths
	Spurious: Forcing both paths & Forcing 1 particular path
	Labels: 0-1000 Good 1000-1500 Bad 1500-2000 Bad2 		
	0-150 Good 150-225 Bad 200-225 Bad2 225-300 

Dataset5: Delayed goals, visit first something and until that, you can not visit another place
	Spurious: Being able to visit the other place whenever you want
	Labels: 0-1000 Good 1000-2000 Bad 		
	0-150 Good 150-300 Bad 


Dataset6: Mutual exclusive paths, visit either one or two but never both.
	 Spurious: Visit both
	Labels: 0-1000 Good 1000-2000 Bad 		
		0-150 Good 150-300 Bad

Dataset7.1: Recurrent paths, do a certain path 2 times
	Spurious: Second round is done in reverse
	Labels: 0-1000 Good 1000-2000 Bad 		
	0-150 Good 150-225 Bad 200-300 

Dataset7.2: Recurrent paths, do a certain path 2 times
	Spurious: There is no secondTime
	Labels: 0-1000 Good 1000-2000 Bad 		
	0-150 Good 150-225 Bad 200-300 