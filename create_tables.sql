CREATE TABLE IF NOT EXISTS `user` (
  `id` INT NOT NULL AUTO_INCREMENT COMMENT 'Auto Increase',
  `password` VARCHAR(255) NOT NULL COMMENT '암호화된 SSO 비밀번호',
  `nickname` VARCHAR(255) NOT NULL COMMENT '로그인 ID 또는 닉네임',
  `eXSignOnSessionID` VARCHAR(255) NOT NULL COMMENT '세션ID',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_user_nickname` (`nickname`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `aiot_notices` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `json_data` JSON NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
