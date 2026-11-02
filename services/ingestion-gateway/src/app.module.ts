import { Module } from '@nestjs/common';
import { HealthController } from './health.controller';
import { TelemetryModule } from './telemetry/telemetry.module';

@Module({
  imports: [TelemetryModule],
  controllers: [HealthController],
})
export class AppModule {}
