import { Inject, Injectable, Logger, OnModuleDestroy, OnModuleInit, ServiceUnavailableException } from '@nestjs/common';
import { ClientKafka } from '@nestjs/microservices';
import { lastValueFrom, timeout } from 'rxjs';
import { KAFKA_CLIENT, RAW_SIGNALS_TOPIC } from './telemetry.constants';
import { TelemetryChunkDto } from './telemetry.dto';

@Injectable()
export class TelemetryService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(TelemetryService.name);

  constructor(@Inject(KAFKA_CLIENT) private readonly kafka: ClientKafka) {}

  async onModuleInit(): Promise<void> {
    await this.kafka.connect();
    this.logger.log(`Kafka producer connected, topic=${RAW_SIGNALS_TOPIC}`);
  }

  async onModuleDestroy(): Promise<void> {
    await this.kafka.close();
  }

  /** Publishes the raw chunk, keyed by patientId so each patient's data stays ordered on one partition. */
  async publish(chunk: TelemetryChunkDto): Promise<void> {
    try {
      await lastValueFrom(
        this.kafka
          .emit(RAW_SIGNALS_TOPIC, { key: chunk.patientId, value: JSON.stringify(chunk) })
          .pipe(timeout(5000)),
      );
      this.logger.log(`published patient=${chunk.patientId} ts=${chunk.timestamp} samples=${chunk.signalVector.length}`);
    } catch (err) {
      this.logger.error(`Kafka publish failed: ${(err as Error).message}`);
      throw new ServiceUnavailableException('Telemetry pipeline temporarily unavailable');
    }
  }
}
