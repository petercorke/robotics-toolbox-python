# ======================================================================== #


if __name__ == "__main__":
    map = LandmarkMap(20)
    print(map)
    print(map.landmark(2))

    veh = rtb.Bicycle()

    rs = RangeBearingSensor(veh, map)
    print(rs)

    for i in range(10):
        print(rs.reading())

    print(rs.h([0, 0, 0], 10))
    print(rs.h([0, 0, 0], [3, 4]))
