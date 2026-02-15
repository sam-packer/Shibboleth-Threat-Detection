-- Use this if you are starting your data collection with metrics version v1
create table rba_device
(
    device_uuid uuid                                   not null
        primary key,
    first_seen  timestamp with time zone default now() not null,
    last_seen   timestamp with time zone default now() not null
);
