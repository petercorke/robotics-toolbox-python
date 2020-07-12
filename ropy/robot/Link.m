


        
        function display(l)
            %Link.display Display parameters
            %
            % L.display() displays the link parameters in compact single line format.  If L is a
            % vector of Link objects displays one line per element.
            %
            % Notes::
            % - This method is invoked implicitly at the command line when the result
            %   of an expression is a Link object and the command has no trailing
            %   semicolon.
            %
            % See also Link.char, Link.dyn, SerialLink.showlink.
            loose = strcmp( get(0, 'FormatSpacing'), 'loose');
            if loose
                disp(' ');
            end
            disp([inputname(1), ' = '])
            disp( char(l) );
        end % display()
        
        function s = char(links, from_robot)
            %Link.char Convert to string
            %
            % s = L.char() is a string showing link parameters in a compact single line format.
            % If L is a vector of Link objects return a string with one line per Link.
            %
            % See also Link.display.
            
            % display in the order theta d a alpha
            if nargin < 2
                from_robot = false;
            end
            
            s = '';
            
            for j=1:length(links)
                l = links(j);
                
                if l.mdh == 0
                    conv = 'std';
                else
                    conv = 'mod';
                end
                if length(links) == 1
                    qname = 'q';
                else
                    qname = sprintf('q%d', j);
                end
                
                if from_robot
                    fmt = '%11g';
                    % invoked from SerialLink.char method, format for table
                    if l.isprismatic
                        % prismatic joint
                        js = sprintf('|%3d|%11s|%11s|%11s|%11s|%11s|', ...
                            j, ...
                            render(l.theta, fmt), ...
                            qname, ...
                            render(l.a, fmt), ...
                            render(l.alpha, fmt), ...
                            render(l.offset, fmt));
                    else
                        % revolute joint
                        js = sprintf('|%3d|%11s|%11s|%11s|%11s|%11s|', ...
                            j, ...
                            qname, ...
                            render(l.d, fmt), ...
                            render(l.a, fmt), ...
                            render(l.alpha, fmt), ...
                            render(l.offset, fmt));
                    end
                else
                    if length(links) == 1
                       if l.isprismatic
                        % prismatic joint
                        js = sprintf('Prismatic(%s): theta=%s, d=%s, a=%s, alpha=%s, offset=%s', ...
                            conv, ...
                            render(l.theta,'%g'), ...
                            qname, ...
                            render(l.a,'%g'), ...
                            render(l.alpha,'%g'), ...
                            render(l.offset,'%g') );
                    else
                        % revolute
                        js = sprintf('Revolute(%s): theta=%s, d=%s, a=%s, alpha=%s, offset=%s', ...
                            conv, ...
                            qname, ...
                            render(l.d,'%g'), ...
                            render(l.a,'%g'), ...
                            render(l.alpha,'%g'), ...
                            render(l.offset,'%g') );
                    end
                    else
                    if l.isprismatic
                        % prismatic joint
                        js = sprintf('Prismatic(%s): theta=%s   d=%s a=%s alpha=%s offset=%s', ...
                            conv, ...
                            render(l.theta), ...
                            qname, ...
                            render(l.a), ...
                            render(l.alpha), ...
                            render(l.offset) );
                    else
                        % revolute
                        js = sprintf('Revolute(%s):  theta=%s   d=%s a=%s alpha=%s offset=%s', ...
                            conv, ...
                            qname, ...
                            render(l.d), ...
                            render(l.a), ...
                            render(l.alpha), ...
                            render(l.offset) );
                    end
                    end
                end
                if isempty(s)
                    s = js;
                else
                    s = char(s, js);
                end
            end
            

        end % char()
        
        function dyn(links)
            %Link.dyn Show inertial properties of link
            %
            % L.dyn() displays the inertial properties of the link object in a multi-line
            % format. The properties shown are mass, centre of mass, inertia, friction,
            % gear ratio and motor properties.
            %
            % If L is a vector of Link objects show properties for each link.
            %
            % See also SerialLink.dyn.
            
            for j=1:numel(links)
                l = links(j);
                if numel(links) > 1
                    fprintf('\nLink %d::', j);
                end
                fprintf('%s\n', l.char());
                if ~isempty(l.m)
                    fprintf('  m    = %s\n', render(l.m))
                end
                if ~isempty(l.r)
                    s = render(l.r);
                    fprintf('  r    = %s %s %s\n', s{:});
                end
                if ~isempty(l.I)
                    s = render(l.I(1,:));
                    fprintf('  I    = | %s %s %s |\n', s{:});
                    s = render(l.I(2,:));
                    fprintf('         | %s %s %s |\n', s{:});
                    s = render(l.I(3,:));
                    fprintf('         | %s %s %s |\n', s{:});
                end
                if ~isempty(l.Jm)
                    fprintf('  Jm   = %s\n', render(l.Jm));
                end
                if ~isempty(l.B)
                    fprintf('  Bm   = %s\n', render(l.B));
                end
                if ~isempty(l.Tc)
                    fprintf('  Tc   = %s(+) %s(-)\n', ...
                        render(l.Tc(1)), render(l.Tc(2)));
                end
                if ~isempty(l.G)
                    fprintf('  G    = %s\n', render(l.G));
                end
                if ~isempty(l.qlim)
                    fprintf('  qlim = %f to %f\n', l.qlim(1), l.qlim(2));
                end
            end
        end % dyn()
