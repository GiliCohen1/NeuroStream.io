import { Module } from '@nestjs/common';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { KAFKA_CLIENT } from './telemetry.constants';
import { TelemetryController } from './telemetry.controller';
import { TelemetryService } from './telemetry.service';

@Module({
  imports: [
    ClientsModule.register([
      {
        name: KAFKA_CLIENT,
        transport: Transport.KAFKA,
        options: {
          client: {
            clientId: 'ingestion-gateway',
            brokers: (process.env.KAFKA_BROKERS ?? 'localhost:29092').split(','),
            retry: { retries: 8, initialRetryTime: 300 },
          },
          producer: { allowAutoTopicCreation: false },
          producerOnlyMode: true,
        },
      },
    ]),
  ],
  controllers: [TelemetryController],
  providers: [TelemetryService],
})
export class TelemetryModule {}
