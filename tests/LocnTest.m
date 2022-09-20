function tests = LocnTest
  tests = functiontests(localfunctions);
  clc
end

function setupOnce(testCase)
    testCase.TestData.Duration = 50;
end

function Vehicle_test(tc)
    %%
    randinit
    V = diag([0.005, 0.5*pi/180].^2);

    v = Bicycle('covar', V);
    v.add_driver( RandomPath(10) );

    v.run(tc.TestData.Duration);
    v.plot_xy();
    s = v.char();

    J = v.Fx(v.x, [.1 .2]);
    J = v.Fv(v.x, [.1 .2]);
end

function DeadReckoning_test(tc)
    %%
    randinit
    V = diag([0.005, 0.5*pi/180].^2);
    P0 = diag([0.005, 0.005, 0.001].^2);

    v = Bicycle('covar', V);
    v.add_driver( RandomPath(10) );
    s = char(v);

    ekf = EKF(v, V, P0);
    ekf.run(tc.TestData.Duration);

    clf
    ekf.plot_xy
    hold on 
    v.plot_xy('r')
    grid on
    xyzlabel

    ekf.plot_ellipse('g')
    ekf.plot_P()
end

function MapLocalization_test(tc)
    %%
    randinit
    W = diag([0.1, 1*pi/180].^2);
    P0 = diag([0.005, 0.005, 0.001].^2);
    V = diag([0.005, 0.5*pi/180].^2);

    map = LandmarkMap(20);
    map = LandmarkMap(20, 'verbose');
    map = LandmarkMap(20, 10, 'verbose');
    map = LandmarkMap(20, 10);
    s = char(map);

    veh = Bicycle('covar', V);
    veh.add_driver( RandomPath(10) );
    sensor = RangeBearingSensor(veh, map, 'covar', W);
    sensor.interval = 5;
    ekf = EKF(veh, W, P0, sensor, W, map);

    ekf.run(tc.TestData.Duration);

    clf
    map.plot()
    veh.plot_xy('b');
    ekf.plot_xy('r');
    ekf.plot_ellipse('k')
    grid on
    xyzlabel

    clf
    ekf.plot_P()
end

function Mapping_test(tc)
    %%
    randinit
    W = diag([0.1, 1*pi/180].^2);
    V = diag([0.005, 0.5*pi/180].^2);

    map = LandmarkMap(20, 10);

    veh = Bicycle('covar', V);
    veh.add_driver( RandomPath(10) );

    sensor = RangeBearingSensor(veh, map, 'covar', W);
    sensor.interval = 5;

    ekf = EKF(veh, [], [], sensor, W, []);
    ekf.run(tc.TestData.Duration);
    

    clf
    map.plot()
    veh.plot_xy('b');
    ekf.plot_map('g');
    grid on
    xyzlabel
    
    %%
    verifyEqual(tc, numcols(ekf.landmarks), 20);

end

function SLAM_test(tc)
    %%
    randinit
    W = diag([0.1, 1*pi/180].^2);
    P0 = diag([0.005, 0.005, 0.001].^2);
    V = diag([0.005, 0.5*pi/180].^2);

    map = LandmarkMap(20, 10);

    veh = Bicycle(V);
    veh.add_driver( RandomPath(10) );

    sensor = RangeBearingSensor(veh, map, 'covar', W);
    sensor.interval = 1;

    ekf = EKF(veh, V, P0, sensor, W, []);
    ekf
    ekf.verbose = false;
    ekf.run(tc.TestData.Duration);


    clf
    map.plot()
    veh.plot_xy('b');
    ekf.plot_xy('r');
    ekf.plot_ellipse('k')
    grid on
    xyzlabel

    clf
    ekf.plot_P()

    clf
    map.plot();
    ekf.plot_map('g');
    
    %%
    verifyEqual(tc, numcols(ekf.landmarks), 20);

end

function ParticleFilter_test(tc)
    %%
    randinit
    map = LandmarkMap(20);

    W = diag([0.1, 1*pi/180].^2);
    v = Bicycle('covar', W);
    v.add_driver( RandomPath(10) );
    V = diag([0.005, 0.5*pi/180].^2);
    sensor = RangeBearingSensor(v, map, 'covar', V);

    Q = diag([0.1, 0.1, 1*pi/180]).^2;
    L = diag([0.1 0.1]);
    pf = ParticleFilter(v, sensor, Q, L, 1000);
    pf
    pf.run(tc.TestData.Duration);

    plot(pf.std)
    xlabel('time step')
    ylabel('standard deviation')
    legend('x', 'y', '\theta')
    grid       

    clf
    pf.plot_pdf();
    clf
    pf.plot_xy();
end

function posegraph_test(tc)
    pg = PoseGraph('pg1.g2o')
    tc.verifyClass(pg, 'PoseGraph');
    tc.verifyEqual(pg.graph.n, 4);
    
    clf
    pg.plot()
    pg.optimize('animate')
    close all
    
    pg = PoseGraph('killian-small.toro')
    tc.verifyClass(pg, 'PoseGraph');
    tc.verifyEqual(pg.graph.n, 1941);
    
    pg = PoseGraph('killian.g2o', 'laser')
    tc.verifyClass(pg, 'PoseGraph');
    tc.verifyEqual(pg.graph.n, 3873);
    
    [r,theta] = pg.scan(1);
    tc.verifyClass(r, 'double');
    tc.verifyLength(r, 180);
    tc.verifyClass(theta, 'double');
    tc.verifyLength(theta, 180);
    
    [x,y] = pg.scan(1);
    tc.verifyClass(x, 'double');
    tc.verifyLength(x, 180);
    tc.verifyClass(y, 'double');
    tc.verifyLength(y, 180);
    
    pose = pg.pose(1);
    tc.verifyClass(pose, 'double');
    tc.verifySize(pose, [3 1]);
    
    t = pg.time(1);
    tc.verifyClass(t, 'double');
    tc.verifySize(t, [1 1]);
    
    w = pg.scanmap('ngrid', 3000);
    tc.verifyClass(w, 'int32');
    tc.verifySize(w, [3000 3000]);
    
    tc.assumeTrue(exist('idisp', 'file'));  %REMINDER
    clf
    pg.plot_occgrid(w);
    close all
    
end

function makemap_test(tc)
        tc.assumeTrue(false);  %REMINDER

end

function chi2inv_test(tc)
    tc.verifyEqual( chi2inv_rtb(0,2), 0);
    tc.verifyEqual( chi2inv_rtb(1,2), Inf);
    tc.verifyEqual( chi2inv_rtb(3,2), NaN);
    
    tc.verifyError( @() chi2inv_rtb(1,1), 'RTB:chi2inv_rtb:badarg');
    
end