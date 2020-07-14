


        function s = char(robot)
        %SerialLink.char Convert to string
        %
        % S = R.char() is a string representation of the robot's kinematic parameters,
        % showing DH parameters, joint structure, comments, gravity vector, base and 
        % tool transform.

            s = '';
            for j=1:length(robot)
                r = robot(j);

                % informational line
                info = '';
                if r.mdh
                    info = strcat(info, 'modDH');
                else
                    info = strcat(info, 'stdDH');
                end
                if r.fast
                    info = strcat(info, ', fastRNE');
                else
                    info = strcat(info, ', slowRNE');
                end
                if r.issym
                    info = strcat(info, ', Symbolic');;
                end
                manuf = '';
                if ~isempty(r.manufacturer)
                    manuf = [' [' r.manufacturer ']'];
                end

                s = sprintf('%s%s:: %d axis, %s, %s', r.name, manuf, r.n, r.config, info);

                % comment and other info
                line = '';
%                 if ~isempty(r.manufacturer)
%                     line = strcat(line, sprintf(' from %s;', r.manufacturer));
%                 end
                if ~isempty(r.comment)
                    line = strcat(line, sprintf(' - %s;', r.comment));
                end
                if ~isempty(line)
                s = char(s, line);
                end

                % link parameters
                s = char(s, '+---+-----------+-----------+-----------+-----------+-----------+');
                s = char(s, '| j |     theta |         d |         a |     alpha |    offset |');
                s = char(s, '+---+-----------+-----------+-----------+-----------+-----------+');
                s = char(s, char(r.links, true));
                s = char(s, '+---+-----------+-----------+-----------+-----------+-----------+');


                % gravity, base, tool
%                s_grav = horzcat(char('grav = ', ' ', ' '), mat2str(r.gravity'));
%                 s_grav = char(s_grav, ' ');
%                 s_base = horzcat(char('  base = ',' ',' ', ' '), mat2str(r.base));
% 
%                 s_tool = horzcat(char('   tool =  ',' ',' ', ' '), mat2str(r.tool));
% 
%                 line = horzcat(s_grav, s_base, s_tool);
                %s = char(s, sprintf('gravity: (%g, %g, %g)', r.gravity));
                if ~isidentity(r.base)
                    s = char(s, ['base:    ' trprint(r.base.T, 'xyz')]);
                end
                if ~isidentity(r.tool)
                    s = char(s, ['tool:    ' trprint(r.tool.T, 'xyz')]);
                end

                if j ~= length(robot)
                    s = char(s, ' ');
                end
            end
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %   set/get methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function v = get.d(r)
            v = [r.links.d];
        end
        function v = get.a(r)
            v = [r.links.a];
        end
        function v = get.theta(r)
            v = [r.links.theta];
        end
        function v = get.alpha(r)
            v = [r.links.alpha];
        end
        
        function set.offset(r, v)
            if length(v) ~= length(v)
                error('offset vector length must equal number DOF');
            end
            L = r.links;
            for i=1:r.n
                L(i).offset = v(i);
            end
            r.links = L;
        end

        function v = get.offset(r)
            v = [r.links.offset];
        end

        function set.qlim(r, v)
            if numrows(v) ~= r.n
                error('insufficient rows in joint limit matrix');
            end
            L = r.links;
            for i=1:r.n
                L(i).qlim = v(i,:);
            end
            r.links = L;
        end

        function v = get.qlim(r)
            L = r.links;
            v = zeros(r.n, 2);
            for i=1:r.n
                if isempty(L(i).qlim)
                    if L(i).isrevolute
                        v(i,:) = [-pi pi];
                    else
                        v(i,:) = [-Inf Inf];
                    end
                else
                    v(i,:) = L(i).qlim;
                end
            end
        end

        % general methods
                
        function v = configstr(r)
        %SerialLink.configstr
        %
        % V = R.configstr is a string describing the joint types and successive
        % joint relative orientation. The symbol R is used to denote a revolute
        % joint and P for a prismatic joint.  The symbol between these letters
        % indicates whether the adjacent joints are parallel or orthogonal.
        %
        % See also: SerialLink.config.
        
            v = '';
            for i=1:r.n
                v = [v r.links(i).type];
                if i < r.n
                    if r.links(i).alpha == 0
                        v = [v char(hex2dec('007c'))];  % pipe
                    elseif abs(abs(r.links(i).alpha) - pi/2) < 1*eps
                        v = [v char(hex2dec('27c2'))];  % perp
                    else
                        v = [v '?'];
                    end
                end
            end
        end
        




        
        function t = jointdynamics(robot, q, qd)
            %SerialLink.jointdyamics Transfer function of joint actuator
            %
            % TF = R.jointdynamic(Q) is a vector of N continuous-time transfer function
            % objects that represent the transfer function 1/(Js+B) for each joint
            % based on the dynamic parameters of the robot and the configuration Q
            % (1xN).  N is the number of robot joints.
            %
            % % TF = R.jointdynamic(Q, QD) as above but include the linearized effects
            % of Coulomb friction when operating at joint velocity QD (1xN).
            %
            % Notes::
            % - Coulomb friction is ignoredf.
            %
            % See also tf, SerialLink.rne.
            
            for j=1:robot.n
                link = robot.links(j);
                
                % compute inertia for this joint
                zero = zeros(1, robot.n);
                qdd = zero; qdd(j) = 1;
                M = robot.rne(q, zero, qdd, 'gravity', [0 0 0]);
                J = link.Jm + M(j)/abs(link.G)^2;
                
                % compute friction
                B = link.B;
                if nargin == 3
                    % add linearized Coulomb friction at the operating point
                    if qd > 0
                        B = B + link.Tc(1)/qd(j);
                    elseif qd < 0
                        B = B + link.Tc(2)/qd(j);
                    end
                end
                t(j) = tf(1, [J B]);

            end
        end
        
        function jt = jtraj(r, T1, T2, t, varargin)
            %SerialLink.jtraj Joint space trajectory
            %
            % Q = R.jtraj(T1, T2, K, OPTIONS) is a joint space trajectory (KxN) where
            % the joint coordinates reflect motion from end-effector pose T1 to T2 in K
            % steps, where N is the number of robot joints. T1 and T2 are SE3 objects or homogeneous transformation
            % matrices (4x4). The trajectory Q has one row per time step, and one
            % column per joint.
            %
            % Options::
            % 'ikine',F   A handle to an inverse kinematic method, for example
            %             F = @p560.ikunc.  Default is ikine6s() for a 6-axis spherical
            %             wrist, else ikine().
            %
            % Notes::
            % - Zero boundary conditions for velocity and acceleration are assumed.
            % - Additional options are passed as trailing arguments to the
            %   inverse kinematic function, eg. configuration options like 'ru'.
            %
            % See also jtraj, SerialLink.ikine, SerialLink.ikine6s.
            if r.isspherical && (r.n == 6)
                opt.ikine = @r.ikine6s;
            else
                opt.ikine = @r.ikine;
            end
            [opt,args] = tb_optparse(opt, varargin);
            
            q1 = opt.ikine(T1, args{:});
            q2 = opt.ikine(T2, args{:});
            
            jt = jtraj(q1, q2, t);
        end
        
        
        function dyn(r, j)
            %SerialLink.dyn Print inertial properties
            %
            % R.dyn() displays the inertial properties of the SerialLink object in a multi-line
            % format.  The properties shown are mass, centre of mass, inertia, gear ratio,
            % motor inertia and motor friction.
            %
            % R.dyn(J) as above but display parameters for joint J only.
            %
            % See also Link.dyn.
            if nargin == 2
                r.links(j).dyn()
            else
                r.links.dyn();
            end
        end
        


                 
        function p = isprismatic(robot)
        %SerialLink.isprismatic identify prismatic joints
        %
        % X = R.isprismatic is a list of logical variables, one per joint, true if
        % the corresponding joint is prismatic, otherwise false.
        %
        % See also Link.isprismatic, SerialLink.isrevolute.
            p = robot.links.isprismatic();
        end
        
        function p = isrevolute(robot)
        %SerialLink.isrevolute identify revolute joints
        %
        % X = R.isrevolute is a list of logical variables, one per joint, true if
        % the corresponding joint is revolute, otherwise false.
        %
        % See also Link.isrevolute, SerialLink.isprismatic.
            p = robot.links.isrevolute();
        end
        
        function qdeg = todegrees(robot, q)
        %SerialLink.todegrees Convert joint angles to degrees
        %
        % Q2 = R.todegrees(Q) is a vector of joint coordinates where those elements
        % corresponding to revolute joints are converted from radians to degrees.
        % Elements corresponding to prismatic joints are copied unchanged.
        %
        % See also SerialiLink.toradians.
            k = robot.isrevolute;
            qdeg = q;
            qdeg(:,k) = qdeg(:,k) * 180/pi;
        end
        
        function qrad = toradians(robot, q)
        %SerialLink.toradians Convert joint angles to radians
        %
        % Q2 = R.toradians(Q) is a vector of joint coordinates where those elements
        % corresponding to revolute joints are converted from degrees to radians.
        % Elements corresponding to prismatic joints are copied unchanged.
        %
        % See also SerialiLink.todegrees.
            k = robot.isrevolute;
            qrad = q;
            qrad(:,k) = qrad(:,k) * pi/180;
        end
        
        function J = jacobn(robot, varargin)
            warning('RTB:SerialLink:deprecated', 'Use jacobe instead of jacobn');
            J = robot.jacobe(varargin{:});
        end
        
        function rmdh = MDH(r)
        %SerialLink.MDH  Convert standard DH model to modified
        %
        % rmdh = R.MDH() is a SerialLink object that represents the same kinematics
        % as R but expressed using modified DH parameters.
        %
        % Notes::
        % - can only be applied to a model expressed with standard DH parameters.
        %
        % See also:  DH
        
            assert(isdh(r), 'RTB:SerialLink:badmodel', 'this method can only be applied to a model with standard DH parameters');
            
            % first joint
            switch r.config(1)
                case 'R'
                    link(1) = Link('modified', 'revolute', ...
                        'd', r.links(1).d, ...
                        'offset', r.links(1).offset, ...
                        'qlim', r.links(1).qlim );
                case 'P'
                    link(1) = Link('modified', 'prismatic', ...
                        'theta', r.links(1).theta, ...
                        'offset', r.links(1).offset, ...
                        'qlim', r.links(1).qlim );
            end

            % middle joints
            for i=2:r.n
                switch r.config(i)
                    case 'R'
                        link(i) = Link('modified', 'revolute', ...
                            'a', r.links(i-1).a, ...
                            'alpha', r.links(i-1).alpha, ...
                            'd', r.links(i).d, ...
                            'offset', r.links(i).offset, ...
                            'qlim', r.links(i).qlim );
                    case 'P'
                        link(i) = Link('modified', 'prismatic', ...
                            'a', r.links(i-1).a, ...
                            'alpha', r.links(i-1).alpha, ...
                            'theta', r.links(i).theta, ...
                            'offset', r.links(i).offset, ...
                            'qlim', r.links(i).qlim );
                end
            end
            
            % last joint
            tool = SE3(r.links(r.n).a, 0, 0) * SE3.Rx(r.links(r.n).alpha) * r.tool;
            
            rmdh = SerialLink(link, 'base', r.base, 'tool', tool);
        end
        
        function rdh = DH(r)
        %SerialLink.DH  Convert modified DH model to standard
        %
        % rmdh = R.DH() is a SerialLink object that represents the same kinematics
        % as R but expressed using standard DH parameters.
        %
        % Notes::
        % - can only be applied to a model expressed with modified DH parameters.
        %
        % See also:  MDH
            
            assert(ismdh(r), 'RTB:SerialLink:badmodel', 'this method can only be applied to a model with modified DH parameters');
            
            base = r.base * SE3(r.links(1).a, 0, 0) * SE3.Rx(r.links(1).alpha);

            % middle joints
            for i=1:r.n-1
                switch r.config(i)
                    case 'R'
                        link(i) = Link('standard', 'revolute', ...
                            'a', r.links(i+1).a, ...
                            'alpha', r.links(i+1).alpha, ...
                            'd', r.links(i).d, ...
                            'offset', r.links(i).offset, ...
                            'qlim', r.links(i).qlim );
                    case 'P'
                        link(i) = Link('standard', 'prismatic', ...
                            'a', r.links(i+1).a, ...
                            'alpha', r.links(i+1).alpha, ...
                            'theta', r.links(i).theta, ...
                            'offset', r.links(i).offset, ...
                            'qlim', r.links(i).qlim );
                end
            end
            
            % last joint
            switch r.config(r.n)
                case 'R'
                    link(r.n) = Link('standard', 'revolute', ...
                        'd', r.links(r.n).d, ...
                        'offset', r.links(r.n).offset, ...
                        'qlim', r.links(r.n).qlim );
                case 'P'
                    link(r.n) = Link('standard', 'prismatic', ...
                        'theta', r.links(r.n).theta, ...
                        'offset', r.links(r.n).offset, ...
                        'qlim', r.links(r.n).qlim );
            end
            
            rdh = SerialLink(link, 'base', base, 'tool', r.tool);
        end
        
        function [tw,T0] = twists(r, q)
        %SerialLink.twists Joint axis twists
        %
        % [TW,T0] = R.twists(Q) is a vector of Twist objects (1xN) that represent
        % the axes of the joints for the robot with joint coordinates Q (1xN).  T0
        % is an SE3 object representing the pose of the tool.
        %
        % [TW,T0] = R.twists() as above but the joint coordinates are taken to be
        % zero.
        %
        % Notes::
        % - [TW,T0] is the product of exponential representation of the robot's
        %   forward kinematics:  prod( [TW.exp(Q) T0] )
        %
        % See also Twist.
            if nargin < 2
                q = zeros(1, r.n);
            end
            
            [Tn,T] = r.fkine( q );
            if r.isdh
                % DH case
                for i=1:r.n
                    if i == 1
                        tw(i) = Twist( r.links(i).type, [0 0 1], [0 0 0]);
                    else
                        tw(i) = Twist( r.links(i).type, T(i-1).a, T(i-1).t);
                    end
                end
            else
                % MDH case
                for i=1:r.n
                    tw(i) = Twist( r.links(i).type, T(i).a, T(i).t);
                end
            end
            
            if nargout > 1
                T0 = Tn;
            end
        end

        
    end % methods

end % classdef
