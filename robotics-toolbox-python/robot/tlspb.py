function [q,qd,qdd] = lspb(q0, q1, t, V)

	if size(t) == [1 1],
		t = [0:t-1]';
	end

	tf = max(t(:));

	if nargin < 4,
		V = (q1-q0)/tf * 1.5;
	else
		if V < (q1-q0)/tf,
			error('V too small\n');
		elseif V > 2*(q1-q0)/tf,
			error('V too big\n');
		end
	end

	tb = (q0 - q1 + V*tf)./V;
	a = V./tb;

	for i = 1:length(t),
		ti = t(i);

		if ti <= tb,
			p(i) = q0 + a/2*ti^2;
			pd(i) = a*ti;
			pdd(i) = a;
		elseif ti <= (tf-tb),
			p(i) = (q1+q0-V*tf)/2 + V*ti;
			pd(i) = V;
			pdd(i) = 0;
		else
			p(i) = q1 - a/2*tf^2 + a*tf*ti - a/2*ti^2;
			pd(i) = -a*ti + a*tf;
			pdd(i) = -a;
		end
	end

	if nargout == 0,
		hold on
		k = t<= tb;
		plot(t(k), p(k), 'r-o');
		k = (t>=tb) & (t<= (tf-tb));
		plot(t(k), p(k), 'b-o');
		k = t>= (tf-tb);
		plot(t(k), p(k), 'g-o');
		grid
		hold off
	else
		q = p';
		qd = pd';
		qdd = pdd';
	end
