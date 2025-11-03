-- Use this if you are starting your data collection with metrics version v4
create table rba_scores
(
    login_id          bigint not null,
    username          text   not null,
    nn_score          double precision,
    ip_risk_score     double precision,
    impossible_travel double precision,
    final_score       double precision,
    created_at        timestamp with time zone default now(),
    primary key (login_id, username),
    constraint fk_rba_scores_login_event
        foreign key (login_id, username) references rba_login_event
            on delete cascade
);