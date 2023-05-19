dbRoot= '';
gtPath= [dbRoot, 'groundtruth/'];

qImagelistFn= [dbRoot, 'groundtruth/querylist.txt'];
qPath= [dbRoot, 'queries_real/'];
dbImagelistFn= [dbRoot, 'groundtruth/imagelist.txt'];
dbPath= [dbRoot, 'images/'];

qImageFns= textscan(fopen(qImagelistFn, 'r'), '%s'); qImageFns= qImageFns{1};
dbImageFns= textscan(fopen(dbImagelistFn, 'r'), '%s'); dbImageFns= dbImageFns{1};

outDir= '~/Relja/Data/Pittsburgh/';

load([gtPath, 'pittsburgh_database_10586_utm.mat'], 'Cdb');
utmDb= reshape(repmat(Cdb,24,1), 2, []); clear Cdb;
load([gtPath, 'pittsburgh_query_1000_utm.mat'], 'Cq');
utmQ= reshape(repmat(Cq,24,1), 2, []); clear Cq;

numImages= length(dbImageFns);
numQueries= length(qImageFns);

% gt lookup
isPosCore= @(utm1, utm2) sum( bsxfun(@minus, utm1, utm2) .^ 2, 1 ) <= 25^2;
isPos= @(iQuery, iDb) isPosCore( utmDb(:,iDb), utmQ(:,iQuery) );