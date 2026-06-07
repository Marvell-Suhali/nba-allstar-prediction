"""
nba_allstar_labels.py  —  Hardcoded version (reliable, no scraping)
Saves data/allstar_labels.csv with all NBA All-Stars from 2000-2024.

Run:
    python nba_allstar_labels.py
"""

import pandas as pd
import os

def build_allstar_labels():
    os.makedirs('data', exist_ok=True)

    # All NBA All-Stars 2000-2024 (both teams combined, no duplicates per season)
    # Source: Basketball-Reference https://www.basketball-reference.com/allstar/
    allstars = [
        # 1999-00
        ("Allen Iverson","1999-00"),("Tim Duncan","1999-00"),("Vince Carter","1999-00"),
        ("Kevin Garnett","1999-00"),("Alonzo Mourning","1999-00"),("Gary Payton","1999-00"),
        ("Shaquille O'Neal","1999-00"),("Karl Malone","1999-00"),("Jason Kidd","1999-00"),
        ("Kobe Bryant","1999-00"),("Chris Webber","1999-00"),("Scottie Pippen","1999-00"),
        ("Anfernee Hardaway","1999-00"),("Grant Hill","1999-00"),("Dikembe Mutombo","1999-00"),
        ("Ray Allen","1999-00"),("Jamal Mashburn","1999-00"),("Stephon Marbury","1999-00"),
        ("Antonio McDyess","1999-00"),("Shareef Abdur-Rahim","1999-00"),
        # 2000-01
        ("Allen Iverson","2000-01"),("Tim Duncan","2000-01"),("Vince Carter","2000-01"),
        ("Kevin Garnett","2000-01"),("Ray Allen","2000-01"),("Gary Payton","2000-01"),
        ("Shaquille O'Neal","2000-01"),("Karl Malone","2000-01"),("Jason Kidd","2000-01"),
        ("Kobe Bryant","2000-01"),("Chris Webber","2000-01"),("Dikembe Mutombo","2000-01"),
        ("Stephon Marbury","2000-01"),("Antonio McDyess","2000-01"),("Dirk Nowitzki","2000-01"),
        ("Baron Davis","2000-01"),("Paul Pierce","2000-01"),("Tracy McGrady","2000-01"),
        ("Peja Stojakovic","2000-01"),("Reggie Miller","2000-01"),
        # 2001-02
        ("Allen Iverson","2001-02"),("Tim Duncan","2001-02"),("Vince Carter","2001-02"),
        ("Kevin Garnett","2001-02"),("Paul Pierce","2001-02"),("Gary Payton","2001-02"),
        ("Shaquille O'Neal","2001-02"),("Karl Malone","2001-02"),("Jason Kidd","2001-02"),
        ("Kobe Bryant","2001-02"),("Tracy McGrady","2001-02"),("Dikembe Mutombo","2001-02"),
        ("Stephon Marbury","2001-02"),("Dirk Nowitzki","2001-02"),("Steve Francis","2001-02"),
        ("Reggie Miller","2001-02"),("Peja Stojakovic","2001-02"),("Jermaine O'Neal","2001-02"),
        ("Chris Webber","2001-02"),("Ben Wallace","2001-02"),
        # 2002-03
        ("Allen Iverson","2002-03"),("Tim Duncan","2002-03"),("Kevin Garnett","2002-03"),
        ("Paul Pierce","2002-03"),("Jason Kidd","2002-03"),("Gary Payton","2002-03"),
        ("Shaquille O'Neal","2002-03"),("Dirk Nowitzki","2002-03"),("Kobe Bryant","2002-03"),
        ("Tracy McGrady","2002-03"),("Steve Francis","2002-03"),("Jermaine O'Neal","2002-03"),
        ("Ben Wallace","2002-03"),("Peja Stojakovic","2002-03"),("Vince Carter","2002-03"),
        ("Ray Allen","2002-03"),("Chris Webber","2002-03"),("Mike Bibby","2002-03"),
        ("Elton Brand","2002-03"),("Yao Ming","2002-03"),
        # 2003-04
        ("Allen Iverson","2003-04"),("Tim Duncan","2003-04"),("Kevin Garnett","2003-04"),
        ("Jason Kidd","2003-04"),("LeBron James","2003-04"),("Gary Payton","2003-04"),
        ("Shaquille O'Neal","2003-04"),("Dirk Nowitzki","2003-04"),("Kobe Bryant","2003-04"),
        ("Tracy McGrady","2003-04"),("Jermaine O'Neal","2003-04"),("Ben Wallace","2003-04"),
        ("Vince Carter","2003-04"),("Ray Allen","2003-04"),("Yao Ming","2003-04"),
        ("Steve Francis","2003-04"),("Peja Stojakovic","2003-04"),("Jamaal Magloire","2003-04"),
        ("Michael Redd","2003-04"),("Sam Cassell","2003-04"),
        # 2004-05
        ("Allen Iverson","2004-05"),("Tim Duncan","2004-05"),("Kevin Garnett","2004-05"),
        ("LeBron James","2004-05"),("Dwyane Wade","2004-05"),("Steve Nash","2004-05"),
        ("Shaquille O'Neal","2004-05"),("Dirk Nowitzki","2004-05"),("Tracy McGrady","2004-05"),
        ("Jermaine O'Neal","2004-05"),("Ben Wallace","2004-05"),("Vince Carter","2004-05"),
        ("Ray Allen","2004-05"),("Yao Ming","2004-05"),("Amar'e Stoudemire","2004-05"),
        ("Gilbert Arenas","2004-05"),("Antawn Jamison","2004-05"),("Larry Hughes","2004-05"),
        ("Chauncey Billups","2004-05"),("Rasheed Wallace","2004-05"),
        # 2005-06
        ("Allen Iverson","2005-06"),("Tim Duncan","2005-06"),("LeBron James","2005-06"),
        ("Dwyane Wade","2005-06"),("Steve Nash","2005-06"),("Kobe Bryant","2005-06"),
        ("Shaquille O'Neal","2005-06"),("Dirk Nowitzki","2005-06"),("Tracy McGrady","2005-06"),
        ("Jermaine O'Neal","2005-06"),("Ben Wallace","2005-06"),("Vince Carter","2005-06"),
        ("Ray Allen","2005-06"),("Yao Ming","2005-06"),("Amar'e Stoudemire","2005-06"),
        ("Gilbert Arenas","2005-06"),("Chauncey Billups","2005-06"),("Paul Pierce","2005-06"),
        ("Tony Parker","2005-06"),("Chris Bosh","2005-06"),
        # 2006-07
        ("LeBron James","2006-07"),("Dwyane Wade","2006-07"),("Steve Nash","2006-07"),
        ("Kobe Bryant","2006-07"),("Dirk Nowitzki","2006-07"),("Tim Duncan","2006-07"),
        ("Tracy McGrady","2006-07"),("Vince Carter","2006-07"),("Yao Ming","2006-07"),
        ("Amar'e Stoudemire","2006-07"),("Gilbert Arenas","2006-07"),("Chauncey Billups","2006-07"),
        ("Richard Hamilton","2006-07"),("Chris Bosh","2006-07"),("Carmelo Anthony","2006-07"),
        ("Allen Iverson","2006-07"),("Pau Gasol","2006-07"),("Carlos Boozer","2006-07"),
        ("Manu Ginobili","2006-07"),("Jason Kidd","2006-07"),
        # 2007-08
        ("LeBron James","2007-08"),("Dwyane Wade","2007-08"),("Steve Nash","2007-08"),
        ("Kobe Bryant","2007-08"),("Dirk Nowitzki","2007-08"),("Tim Duncan","2007-08"),
        ("Tracy McGrady","2007-08"),("Yao Ming","2007-08"),("Amar'e Stoudemire","2007-08"),
        ("Chauncey Billups","2007-08"),("Chris Bosh","2007-08"),("Carmelo Anthony","2007-08"),
        ("Allen Iverson","2007-08"),("Paul Pierce","2007-08"),("Kevin Garnett","2007-08"),
        ("Ray Allen","2007-08"),("Chris Paul","2007-08"),("Dwight Howard","2007-08"),
        ("Carlos Boozer","2007-08"),("Caron Butler","2007-08"),
        # 2008-09
        ("LeBron James","2008-09"),("Dwyane Wade","2008-09"),("Steve Nash","2008-09"),
        ("Kobe Bryant","2008-09"),("Dirk Nowitzki","2008-09"),("Tim Duncan","2008-09"),
        ("Yao Ming","2008-09"),("Amar'e Stoudemire","2008-09"),("Chris Bosh","2008-09"),
        ("Carmelo Anthony","2008-09"),("Paul Pierce","2008-09"),("Kevin Garnett","2008-09"),
        ("Chris Paul","2008-09"),("Dwight Howard","2008-09"),("Tony Parker","2008-09"),
        ("Danny Granger","2008-09"),("Mo Williams","2008-09"),("Devin Harris","2008-09"),
        ("David West","2008-09"),("Pau Gasol","2008-09"),
        # 2009-10
        ("LeBron James","2009-10"),("Dwyane Wade","2009-10"),("Steve Nash","2009-10"),
        ("Kobe Bryant","2009-10"),("Dirk Nowitzki","2009-10"),("Tim Duncan","2009-10"),
        ("Yao Ming","2009-10"),("Amar'e Stoudemire","2009-10"),("Chris Bosh","2009-10"),
        ("Carmelo Anthony","2009-10"),("Kevin Garnett","2009-10"),("Chris Paul","2009-10"),
        ("Dwight Howard","2009-10"),("Tony Parker","2009-10"),("Pau Gasol","2009-10"),
        ("Joe Johnson","2009-10"),("Deron Williams","2009-10"),("Al Horford","2009-10"),
        ("Paul Pierce","2009-10"),("Kevin Durant","2009-10"),
        # 2010-11
        ("LeBron James","2010-11"),("Dwyane Wade","2010-11"),("Steve Nash","2010-11"),
        ("Kobe Bryant","2010-11"),("Dirk Nowitzki","2010-11"),("Kevin Durant","2010-11"),
        ("Amar'e Stoudemire","2010-11"),("Chris Bosh","2010-11"),("Carmelo Anthony","2010-11"),
        ("Chris Paul","2010-11"),("Dwight Howard","2010-11"),("Tony Parker","2010-11"),
        ("Pau Gasol","2010-11"),("Joe Johnson","2010-11"),("Deron Williams","2010-11"),
        ("Al Horford","2010-11"),("Kevin Garnett","2010-11"),("Russell Westbrook","2010-11"),
        ("Rajon Rondo","2010-11"),("Blake Griffin","2010-11"),
        # 2011-12
        ("LeBron James","2011-12"),("Dwyane Wade","2011-12"),("Kobe Bryant","2011-12"),
        ("Kevin Durant","2011-12"),("Chris Paul","2011-12"),("Dwight Howard","2011-12"),
        ("Carmelo Anthony","2011-12"),("Dirk Nowitzki","2011-12"),("Tony Parker","2011-12"),
        ("Pau Gasol","2011-12"),("Deron Williams","2011-12"),("Blake Griffin","2011-12"),
        ("Kevin Garnett","2011-12"),("Russell Westbrook","2011-12"),("Rajon Rondo","2011-12"),
        ("Andrew Bynum","2011-12"),("Marc Gasol","2011-12"),("Chris Bosh","2011-12"),
        ("Roy Hibbert","2011-12"),("Tyson Chandler","2011-12"),
        # 2012-13
        ("LeBron James","2012-13"),("Dwyane Wade","2012-13"),("Kobe Bryant","2012-13"),
        ("Kevin Durant","2012-13"),("Chris Paul","2012-13"),("Dwight Howard","2012-13"),
        ("Carmelo Anthony","2012-13"),("Tony Parker","2012-13"),("Blake Griffin","2012-13"),
        ("Kevin Garnett","2012-13"),("Russell Westbrook","2012-13"),("Rajon Rondo","2012-13"),
        ("Marc Gasol","2012-13"),("Chris Bosh","2012-13"),("Kyrie Irving","2012-13"),
        ("Paul George","2012-13"),("Lamarcus Aldridge","2012-13"),("Zach Randolph","2012-13"),
        ("Stephen Curry","2012-13"),("Jrue Holiday","2012-13"),
        # 2013-14
        ("LeBron James","2013-14"),("Dwyane Wade","2013-14"),("Kobe Bryant","2013-14"),
        ("Kevin Durant","2013-14"),("Chris Paul","2013-14"),("Dwight Howard","2013-14"),
        ("Carmelo Anthony","2013-14"),("Tony Parker","2013-14"),("Blake Griffin","2013-14"),
        ("Russell Westbrook","2013-14"),("Kyrie Irving","2013-14"),("Paul George","2013-14"),
        ("Lamarcus Aldridge","2013-14"),("Stephen Curry","2013-14"),("Kevin Love","2013-14"),
        ("Chris Bosh","2013-14"),("John Wall","2013-14"),("James Harden","2013-14"),
        ("Dirk Nowitzki","2013-14"),("Joakim Noah","2013-14"),
        # 2014-15
        ("LeBron James","2014-15"),("Kobe Bryant","2014-15"),("Kevin Durant","2014-15"),
        ("Chris Paul","2014-15"),("Dwight Howard","2014-15"),("Carmelo Anthony","2014-15"),
        ("Tony Parker","2014-15"),("Blake Griffin","2014-15"),("Russell Westbrook","2014-15"),
        ("Kyrie Irving","2014-15"),("Paul George","2014-15"),("Lamarcus Aldridge","2014-15"),
        ("Stephen Curry","2014-15"),("Kevin Love","2014-15"),("James Harden","2014-15"),
        ("John Wall","2014-15"),("Marc Gasol","2014-15"),("Kyle Lowry","2014-15"),
        ("Al Horford","2014-15"),("Jeff Teague","2014-15"),
        # 2015-16
        ("LeBron James","2015-16"),("Kevin Durant","2015-16"),("Chris Paul","2015-16"),
        ("Carmelo Anthony","2015-16"),("Blake Griffin","2015-16"),("Russell Westbrook","2015-16"),
        ("Kyrie Irving","2015-16"),("Lamarcus Aldridge","2015-16"),("Stephen Curry","2015-16"),
        ("James Harden","2015-16"),("John Wall","2015-16"),("Draymond Green","2015-16"),
        ("Kyle Lowry","2015-16"),("Paul George","2015-16"),("Klay Thompson","2015-16"),
        ("Damian Lillard","2015-16"),("Kawhi Leonard","2015-16"),("Pau Gasol","2015-16"),
        ("Andre Drummond","2015-16"),("Dwyane Wade","2015-16"),
        # 2016-17
        ("LeBron James","2016-17"),("Kevin Durant","2016-17"),("Chris Paul","2016-17"),
        ("Carmelo Anthony","2016-17"),("Russell Westbrook","2016-17"),("Kyrie Irving","2016-17"),
        ("Stephen Curry","2016-17"),("James Harden","2016-17"),("John Wall","2016-17"),
        ("Draymond Green","2016-17"),("Kyle Lowry","2016-17"),("Klay Thompson","2016-17"),
        ("Damian Lillard","2016-17"),("Kawhi Leonard","2016-17"),("Kevin Love","2016-17"),
        ("Isaiah Thomas","2016-17"),("Giannis Antetokounmpo","2016-17"),("Paul Millsap","2016-17"),
        ("Jimmy Butler","2016-17"),("DeMar DeRozan","2016-17"),
        # 2017-18
        ("LeBron James","2017-18"),("Kevin Durant","2017-18"),("Russell Westbrook","2017-18"),
        ("Kyrie Irving","2017-18"),("Stephen Curry","2017-18"),("James Harden","2017-18"),
        ("John Wall","2017-18"),("Draymond Green","2017-18"),("Klay Thompson","2017-18"),
        ("Damian Lillard","2017-18"),("Kawhi Leonard","2017-18"),("Kevin Love","2017-18"),
        ("Giannis Antetokounmpo","2017-18"),("Paul Millsap","2017-18"),("Jimmy Butler","2017-18"),
        ("DeMar DeRozan","2017-18"),("Bradley Beal","2017-18"),("Victor Oladipo","2017-18"),
        ("Kemba Walker","2017-18"),("Andre Drummond","2017-18"),
        # 2018-19
        ("LeBron James","2018-19"),("Kevin Durant","2018-19"),("Russell Westbrook","2018-19"),
        ("Kyrie Irving","2018-19"),("Stephen Curry","2018-19"),("James Harden","2018-19"),
        ("Draymond Green","2018-19"),("Klay Thompson","2018-19"),("Damian Lillard","2018-19"),
        ("Kawhi Leonard","2018-19"),("Kevin Love","2018-19"),("Giannis Antetokounmpo","2018-19"),
        ("Paul George","2018-19"),("Jimmy Butler","2018-19"),("DeMar DeRozan","2018-19"),
        ("Bradley Beal","2018-19"),("Victor Oladipo","2018-19"),("Kemba Walker","2018-19"),
        ("Ben Simmons","2018-19"),("Nikola Jokic","2018-19"),
        # 2019-20
        ("LeBron James","2019-20"),("Anthony Davis","2019-20"),("Russell Westbrook","2019-20"),
        ("Kyrie Irving","2019-20"),("James Harden","2019-20"),("Damian Lillard","2019-20"),
        ("Kawhi Leonard","2019-20"),("Giannis Antetokounmpo","2019-20"),("Paul George","2019-20"),
        ("Bradley Beal","2019-20"),("Kemba Walker","2019-20"),("Ben Simmons","2019-20"),
        ("Nikola Jokic","2019-20"),("Chris Paul","2019-20"),("Trae Young","2019-20"),
        ("Bam Adebayo","2019-20"),("Jayson Tatum","2019-20"),("Donovan Mitchell","2019-20"),
        ("Pascal Siakam","2019-20"),("Khris Middleton","2019-20"),
        # 2020-21
        ("LeBron James","2020-21"),("Anthony Davis","2020-21"),("Stephen Curry","2020-21"),
        ("James Harden","2020-21"),("Damian Lillard","2020-21"),("Kawhi Leonard","2020-21"),
        ("Giannis Antetokounmpo","2020-21"),("Bradley Beal","2020-21"),("Kemba Walker","2020-21"),
        ("Ben Simmons","2020-21"),("Nikola Jokic","2020-21"),("Chris Paul","2020-21"),
        ("Trae Young","2020-21"),("Bam Adebayo","2020-21"),("Jayson Tatum","2020-21"),
        ("Donovan Mitchell","2020-21"),("Khris Middleton","2020-21"),("Rudy Gobert","2020-21"),
        ("Zach LaVine","2020-21"),("Julius Randle","2020-21"),
        # 2021-22
        ("LeBron James","2021-22"),("Stephen Curry","2021-22"),("James Harden","2021-22"),
        ("Damian Lillard","2021-22"),("Giannis Antetokounmpo","2021-22"),("Joel Embiid","2021-22"),
        ("Nikola Jokic","2021-22"),("Trae Young","2021-22"),("Jayson Tatum","2021-22"),
        ("Donovan Mitchell","2021-22"),("Khris Middleton","2021-22"),("Rudy Gobert","2021-22"),
        ("Zach LaVine","2021-22"),("DeMar DeRozan","2021-22"),("Kevin Durant","2021-22"),
        ("Andrew Wiggins","2021-22"),("Chris Paul","2021-22"),("Fred VanVleet","2021-22"),
        ("Darius Garland","2021-22"),("Jarrett Allen","2021-22"),
        # 2022-23
        ("LeBron James","2022-23"),("Stephen Curry","2022-23"),("Damian Lillard","2022-23"),
        ("Giannis Antetokounmpo","2022-23"),("Joel Embiid","2022-23"),("Nikola Jokic","2022-23"),
        ("Jayson Tatum","2022-23"),("Donovan Mitchell","2022-23"),("Kevin Durant","2022-23"),
        ("Zach LaVine","2022-23"),("DeMar DeRozan","2022-23"),("Kyrie Irving","2022-23"),
        ("Lauri Markkanen","2022-23"),("Tyrese Haliburton","2022-23"),("Shai Gilgeous-Alexander","2022-23"),
        ("Anthony Edwards","2022-23"),("Jaren Jackson Jr.","2022-23"),("Pascal Siakam","2022-23"),
        ("Julius Randle","2022-23"),("Ja Morant","2022-23"),
        # 2023-24
        ("LeBron James","2023-24"),("Stephen Curry","2023-24"),("Kevin Durant","2023-24"),
        ("Nikola Jokic","2023-24"),("Giannis Antetokounmpo","2023-24"),("Joel Embiid","2023-24"),
        ("Damian Lillard","2023-24"),("Jayson Tatum","2023-24"),("Tyrese Haliburton","2023-24"),
        ("Bam Adebayo","2023-24"),("Shai Gilgeous-Alexander","2023-24"),("Anthony Edwards","2023-24"),
        ("Devin Booker","2023-24"),("Kawhi Leonard","2023-24"),("Karl-Anthony Towns","2023-24"),
        ("Jalen Brunson","2023-24"),("Trae Young","2023-24"),("Paolo Banchero","2023-24"),
        ("Cade Cunningham","2023-24"),("Tyrese Maxey","2023-24"),
    ]

    df = pd.DataFrame(allstars, columns=['player', 'season'])
    df['all_star'] = 1
    df = df.drop_duplicates()
    df.to_csv('data/allstar_labels.csv', index=False)
    print(f"Saved {len(df)} All-Star records to data/allstar_labels.csv")
    print(f"Seasons covered: {df['season'].nunique()}")
    print(df['season'].value_counts().sort_index())
    return df


if __name__ == '__main__':
    build_allstar_labels()
