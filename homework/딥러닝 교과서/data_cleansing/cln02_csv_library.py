import csv

# with 문을 사용해서 파일을 처리
with open("csv0.csv", "w")as csvfile:
    # writer() 메서드의 인수로 csvfile과 개행 코드(\n)를 지정
    writer = csv.writer(csvfile, lineterminator="\n")

    # writerow(리스트)로 행을 추가
    writer.writerow(["city", "year", 'season'])
    writer.writerow(["Nagano", "1998", 'winter'])
    writer.writerow(["Sydney", "2000", 'summer'])
    writer.writerow(["Salt Lake city", "2002", 'winter'])
    writer.writerow(["Athens", "2004", 'summer'])
    writer.writerow(["Torino", "2006", 'winter'])
    writer.writerow(["Beijing", "2008", 'summer'])
    writer.writerow(["Vancouver", "2010", 'wimter'])
    writer.writerow(["London", "2012", 'summer'])
    writer.writerow(["Sorchi", "2014", 'winter'])
    writer.writerow(["Rio de Janeiro", "2016", 'summer'])
