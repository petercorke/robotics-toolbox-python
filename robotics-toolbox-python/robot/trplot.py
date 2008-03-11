function trplot(T, name, color)

	if nargin == 1,
		fmt = '%c';
	else
		fmt = sprintf('%%c%s', name);
	end
	if nargin < 3,
		color = 'b';
	end

	q = quaternion(T);
	plot(q, T(1:3,4), fmt, color);
